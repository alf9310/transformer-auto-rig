"""
Evaluation of RigAnything on Articulation-XL2.0.

Metrics:
  J2J:  mean Joint-to-Joint distance (Hungarian matching)
  J2B:  mean distance from each predicted joint to nearest GT bone segment
  B2B:  symmetric Chamfer distance between predicted and GT bone midpoints
  IOU / Precision / Recall: macro-averaged per-bone hard skinning assignment

Usage:
    python evaluate_riganything.py \
        --config       config.yaml               \
        --ckpt         ckpt/riganything_ckpt.pt  \
        --dataset_npz  articulation_xlv2_test.npz \
        --metadata_csv articulation_xlv2_metadata.csv \
        --output_csv   results.csv               \
        [--category    human]                    \
        [--device      cuda:0]
"""

from __future__ import annotations

import argparse
import importlib
import os.path as osp
from typing import Dict, Generator, List, Optional, Tuple

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
import torch.nn.functional as F
import trimesh
import yaml
from easydict import EasyDict as edict
from scipy.optimize import linear_sum_assignment
from tqdm import tqdm
import wandb

from utils.job_checkpointer import get_job_overview, resume_job
from utils.optimizer_scheduler import configure_lr_scheduler, configure_optimizer

# NOTE: Big mistake to import from inference.py bc its argparse block lives at
# module level (outside __main__), which would hijack sys.argv on import...

def smooth_weights_per_vertex(mesh, weights, iterations=5, neighbor_factor=0.3):
    """
    Smooth weights using neighbour averaging.
    Copied from inference.py to avoid triggering its module-level
    argparse block on import
    """
    from copy import deepcopy

    smoothed = deepcopy(weights)
    vertex_neighbors = mesh.vertex_neighbors
    for _ in range(iterations):
        new_w = deepcopy(smoothed)
        for i in range(len(mesh.vertices)):
            nbrs = vertex_neighbors[i]
            if len(nbrs) > 0:
                new_w[i] = (1.0 - neighbor_factor) * smoothed[
                    i
                ] + neighbor_factor * np.mean(smoothed[nbrs], axis=0)
                s = np.sum(new_w[i])
                if s > 0:
                    new_w[i] /= s
        smoothed = new_w
    return smoothed


def project_to_glb(glb_vert, mesh_vert, mesh_skinning):
    """
    For each GLB vertex find the closest mesh vertex and transfer its skinning
    Copied from inference.py
    """
    glb_vert = np.asarray(glb_vert, dtype=np.float32)
    mesh_vert = np.asarray(mesh_vert, dtype=np.float32)
    mesh_skinning = np.asarray(mesh_skinning, dtype=np.float32)
    out = []
    for start in range(0, len(glb_vert), 4096):
        batch = glb_vert[start : start + 4096]
        dist = np.linalg.norm(batch[:, None] - mesh_vert[None], axis=-1)
        out.append(mesh_skinning[dist.argmin(axis=1)])
    return np.concatenate(out, axis=0)


# ---------------------------------------------------------------------------
# 1.  Dataset loading (single packed NPZ)
# ---------------------------------------------------------------------------


def _unpack_arr0(npz_path: str):
    """
    Detects the structure of an .npz and returns a list of
    per-sample dicts

    Returns (samples: list[dict], field_names: list[str]) or (None, None)
    if arr_0 is not present / not in a recognised format.
    """
    data = np.load(npz_path, allow_pickle=True)
    if "arr_0" not in data.files:
        return None, None

    arr = data["arr_0"]

    # 0-d object wrapping a single dict  {field: [sample0, sample1, ...]}
    if arr.ndim == 0:
        obj = arr.item()
        if isinstance(obj, dict):
            fields = list(obj.keys())
            n = len(next(iter(obj.values())))
            samples = [{f: obj[f][i] for f in fields} for i in range(n)]
            return samples, fields
        if isinstance(obj, (list, tuple)) and len(obj) > 0 and isinstance(obj[0], dict):
            fields = list(obj[0].keys())
            return list(obj), fields

    # 1-d object array of per-sample dicts
    if arr.ndim == 1 and arr.dtype == object:
        e = arr[0]
        if isinstance(e, dict):
            fields = list(e.keys())
            return list(arr), fields

    # structured array
    if arr.dtype.names:
        fields = list(arr.dtype.names)
        samples = [{f: arr[f][i] for f in fields} for i in range(len(arr))]
        return samples, fields

    return None, None


def inspect_dataset_npz(npz_path: str) -> None:
    """
    HELPER: Print the structure of the NPZ for inspection
    """
    data = np.load(npz_path, allow_pickle=True)
    print(f"\n=== Top-level keys in '{osp.basename(npz_path)}' ===")
    for key in sorted(data.files):
        arr = data[key]
        elem_info = ""
        if arr.dtype == object and arr.ndim >= 1 and len(arr) > 0:
            e = arr[0]
            if hasattr(e, "shape"):
                elem_info = f"  [elem0 dtype={e.dtype} shape={e.shape}]"
            else:
                elem_info = f"  [elem0 type={type(e).__name__} val={repr(e)[:60]}]"
        print(f"  {key:<42s} dtype={arr.dtype}  shape={arr.shape}{elem_info}")

    samples, fields = _unpack_arr0(npz_path)
    if samples is not None:
        print(f"\n=== arr_0 unpacked: {len(samples)} samples, fields: {fields} ===")
        s0 = samples[0]
        for f in fields:
            v = np.asarray(s0[f]) if not isinstance(s0[f], np.ndarray) else s0[f]
            print(f"  {f:<42s} dtype={v.dtype}  shape={v.shape}")
    print()


def iter_dataset_npz(
    npz_path: str,
    metadata_csv: Optional[str] = None,
    category_filter: Optional[str] = None,
) -> Generator[Dict, None, None]:
    """
    Yield one dict per sample from the single packed Articulation-XL2.0 NPZ.
    Key names are matched via a priority alias list for compatability

    Yielded dict keys:
        uuid, vertices [V,3], faces [F,3], normals [V,3],
        joints [J,3], bones_adj [J,J] bool,
        skinning_weights [V,J] float32,
        root_index int, joint_names list[str],
        pc_w_norm [1024,6] float32 or None
    """
    # Optional metadata for category filtering
    uuid_to_meta: Dict[str, dict] = {}
    if metadata_csv is not None and osp.exists(metadata_csv):
        for _, row in pd.read_csv(metadata_csv).iterrows():
            uuid_to_meta[str(row["uuid"])] = row.to_dict()

    # Detect layout
    data = np.load(npz_path, allow_pickle=True)
    top_keys = set(data.files)

    # Try arr_0 packing first
    samples_list, arr0_fields = _unpack_arr0(npz_path)
    use_arr0 = samples_list is not None

    if use_arr0:
        # Normalise field names via alias matching on arr0_fields
        field_set = set(arr0_fields)

        def pick(candidates: List[str], label: str) -> str:
            for c in candidates:
                if c in field_set:
                    return c
            raise KeyError(
                f"Cannot find '{label}' in arr_0 fields {sorted(field_set)}. "
                f"Tried: {candidates}.  Run inspect_dataset_npz() for details."
            )

        source = samples_list  # list of dicts
        n_samples = len(source)
        print(f"Dataset (arr_0 layout): {n_samples} samples")
    else:
        # Per-field top-level keys
        field_set = top_keys

        def pick(candidates: List[str], label: str) -> str:
            for c in candidates:
                if c in field_set:
                    return c
            raise KeyError(
                f"Cannot find '{label}' in NPZ keys {sorted(field_set)}. "
                f"Tried: {candidates}.  Run inspect_dataset_npz() for details."
            )

        source = None  # will index data[key][i] directly
        uuid_arr = data[pick(["uuid", "uuids", "id", "ids", "fileIdentifier"], "uuid")]
        n_samples = len(uuid_arr)
        print(f"Dataset (per-field layout): {n_samples} samples")

    # Resolve field aliases (same list for both layouts)
    k_uuid = pick(["uuid", "uuids", "id", "ids", "fileIdentifier"], "uuid")
    k_verts = pick(["vertices", "verts", "vertex"], "vertices")
    k_faces = pick(["faces", "triangles", "face"], "faces")
    k_norms = pick(["normals", "vertex_normals", "normal"], "normals")
    k_joint = pick(["joints", "joint_positions", "joint"], "joints")
    k_bones = pick(["bones", "bone_matrix", "adjacency", "hier"], "bones")
    k_root = pick(["root_index", "root_idx", "root"], "root_index")
    k_jname = pick(["joint_names", "joint_name", "jnames"], "joint_names")
    k_swv = pick(
        ["skinning_weights_value", "sw_value", "sw_val"], "skinning_weights_value"
    )
    k_swr = pick(["skinning_weights_row", "sw_row"], "skinning_weights_row")
    k_swc = pick(["skinning_weights_col", "sw_col"], "skinning_weights_col")
    k_sws = pick(["skinning_weights_shape", "sw_shape"], "skinning_weights_shape")
    k_pcwn = next(
        (c for c in ["pc_w_norm", "pc_with_normals", "pcwn"] if c in field_set), None
    )

    # Helper: get field i from whichever layout
    def get(key: str, i: int, dtype):
        raw = source[i][key] if use_arr0 else data[key][i]
        return (
            raw.astype(dtype)
            if isinstance(raw, np.ndarray)
            else np.array(raw, dtype=dtype)
        )

    def get_scalar(key: str, i: int):
        raw = source[i][key] if use_arr0 else data[key][i]
        return raw

    # Main loop
    for i in range(n_samples):
        uuid = str(get_scalar(k_uuid, i))

        if category_filter is not None:
            cat = uuid_to_meta.get(uuid, {}).get("category_label", "")
            if category_filter not in cat:
                continue

        vertices = get(k_verts, i, np.float32)  # [V, 3]
        faces = get(k_faces, i, np.int32)  # [F, 3]
        normals = get(k_norms, i, np.float32)  # [V, 3]
        joints = get(k_joint, i, np.float32)  # [J, 3]
        bones_adj = get(k_bones, i, bool)  # [J, J]
        root_index = int(get_scalar(k_root, i))
        joint_names = list(get_scalar(k_jname, i))

        # Reconstruct dense [V, J] skinning weights from COO triplets
        sw_val = get(k_swv, i, np.float32)
        sw_row = get(k_swr, i, np.int32)
        sw_col = get(k_swc, i, np.int32)
        sw_shape = tuple(get(k_sws, i, np.int32).tolist())
        skin_mat = np.asarray(
            sp.coo_matrix((sw_val, (sw_row, sw_col)), shape=sw_shape).todense(),
            dtype=np.float32,
        )  # [V, J]

        pc_w_norm = get(k_pcwn, i, np.float32) if k_pcwn is not None else None

        yield dict(
            uuid=uuid,
            vertices=vertices,
            faces=faces,
            normals=normals,
            joints=joints,
            bones_adj=bones_adj,
            skinning_weights=skin_mat,
            root_index=root_index,
            joint_names=joint_names,
            pc_w_norm=pc_w_norm,
            meta=uuid_to_meta.get(uuid, {}),
        )


# ---------------------------------------------------------------------------
# 2.  RigAnything inference on a GT sample  (extracted from inference.py)
# ---------------------------------------------------------------------------

_N_POINTS = 1024
_TOPK = 5
_SKIN_THRESH = 0.068
_SMOOTH_ITERS = 10
_SMOOTH_FACTOR = 0.35


def _sample_surface_points(
    mesh: trimesh.Trimesh, n: int = _N_POINTS
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Uniformly sample n surface points.
    Mirrors the sampling block in inference.py, plus replacement padding and normal normalisation.
    """
    pts, face_idx = trimesh.sample.sample_surface(mesh, n)
    pts = pts.astype(np.float32)
    nrms = mesh.face_normals[face_idx].astype(np.float32)
    nrms = nrms / np.clip(np.linalg.norm(nrms, axis=1, keepdims=True), 1e-8, None)
    return pts, nrms


def run_inference_on_gt_sample(
    gt_sample: Dict,
    model: torch.nn.Module,
    device: str,
) -> Dict:
    """
    Run RigAnything inference on a single GT sample from iter_dataset_npz.

    Mirrors the per-item inference block in inference.py but reads mesh data
    directly from the gt_sample dict instead of loading a GLB through bpy.

    Does NOT pass full_pointcloud to generate_sequence to prevent KV-cashe overflow

    Returns:
        joints           : [J_pred, 3]  float32  (original coordinate space)
        parents          : [J_pred]     int
        skinning_weights : [V, J_pred]  float32  (projected to all GT vertices)
        pointcloud       : [V, 3]       float32  (GT mesh vertices)
    """
    vertices = gt_sample["vertices"]  # [V, 3]
    faces = gt_sample["faces"]  # [F, 3]
    normals = gt_sample["normals"]  # [V, 3]

    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)

    # 1024-point input cloud
    # Use pc_w_norm only if it is exactly _N_POINTS rows otherwise sample
    pc_w_norm = gt_sample.get("pc_w_norm")
    if pc_w_norm is not None and pc_w_norm.shape[0] == _N_POINTS:
        pts_sampled = pc_w_norm[:, :3].astype(np.float32)
        nrm_sampled = pc_w_norm[:, 3:].astype(np.float32)
    else:
        pts_sampled, nrm_sampled = _sample_surface_points(mesh, _N_POINTS)

    assert (
        pts_sampled.shape[0] == _N_POINTS
    ), f"Expected {_N_POINTS} sampled points, got {pts_sampled.shape[0]}"

    # Normalise to unit sphere (same as inference.py)
    center = (np.max(pts_sampled, axis=0) + np.min(pts_sampled, axis=0)) / 2.0
    pts_norm = (pts_sampled - center) / (np.max(np.abs(pts_sampled - center)) + 1e-8)
    scale = np.max(np.abs(pts_sampled - center))

    # Dummy GT placeholders (required by generate_sequence signature)
    dummy_joints = np.ones((64, 3), dtype=np.float32)
    dummy_parents = np.arange(64, dtype=np.int32)
    dummy_skin = np.zeros((_N_POINTS, 64), dtype=np.float32)

    # NOTE: full_pointcloud is intentionally omitted so generate_sequence uses pc_tokens from the 1024-pt encoding
    batch = {
        "pointcloud": torch.from_numpy(pts_norm).unsqueeze(0).to(device),
        "normals": torch.from_numpy(nrm_sampled).unsqueeze(0).to(device),
        "scale": torch.tensor([scale]).to(device),
        "center": torch.from_numpy(center).to(device),
        "item_idx": [gt_sample["uuid"]],
        "joints": torch.from_numpy(dummy_joints).unsqueeze(0).to(device),
        "parents": torch.from_numpy(dummy_parents).unsqueeze(0).to(device),
        "skinning_weights": torch.from_numpy(dummy_skin).unsqueeze(0).to(device),
        "root_idx": torch.tensor([0]).to(device),
    }

    # Run model (skeleton + skinning on 1024 pts)
    result = model.generate_sequence(
        batch, create_visual=False, save_skeleton=False, compute_loss=False
    )
    npz_dict = result["npz_dict"]

    # Post-process skinning on the 1024 sampled points
    # npz_dict["skinning_weights"] shape: [N_sample, J_pred]  (N_sample = 1024)
    raw_skin = torch.tensor(npz_dict["skinning_weights"])

    k = min(_TOPK, raw_skin.shape[1])
    _, top_idx = torch.topk(raw_skin, k=k, dim=1)
    masked = torch.full_like(raw_skin, -9999.0)
    masked.scatter_(1, top_idx, raw_skin.gather(1, top_idx))
    masked = F.softmax(masked, dim=1)
    masked[masked < _SKIN_THRESH] = 0.0
    skin_norm = masked / (masked.sum(dim=1, keepdim=True) + 1e-6)
    skin_1024 = skin_norm.cpu().numpy()  # [1024, J_pred]

    # Project 1024-pt skinning to all V mesh vertices
    vert_skin = project_to_glb(vertices, pts_sampled, skin_1024)  # [V, J_pred]
    vert_skin /= vert_skin.sum(axis=1, keepdims=True) + 1e-6

    # Smooth over mesh topology
    vert_skin = smooth_weights_per_vertex(
        mesh, vert_skin, iterations=_SMOOTH_ITERS, neighbor_factor=_SMOOTH_FACTOR
    )

    return dict(
        joints=npz_dict["joints"],  # [J_pred, 3]
        parents=npz_dict["parents"],  # [J_pred]
        skinning_weights=vert_skin,  # [V, J_pred]
        pointcloud=vertices,  # [V, 3]
    )


# ---------------------------------------------------------------------------
# 3.  Geometry helpers
# ---------------------------------------------------------------------------


def point_to_segment_distance(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    ab = b - a
    len_sq = float(np.dot(ab, ab))
    if len_sq < 1e-12:
        return float(np.linalg.norm(p - a))
    t = float(np.clip(np.dot(p - a, ab) / len_sq, 0.0, 1.0))
    return float(np.linalg.norm(p - (a + t * ab)))


def parents_to_bones(parents: np.ndarray) -> List[Tuple[int, int]]:
    """Reuses logic from create_armature in vis_skel.py."""
    return [(int(p), int(i)) for i, p in enumerate(parents) if p != i]


def adj_to_bones(bones_adj: np.ndarray) -> List[Tuple[int, int]]:
    rows, cols = np.where(bones_adj)
    return [(int(r), int(c)) for r, c in zip(rows, cols)]


# ---------------------------------------------------------------------------
# 4.  Skeleton metrics
# ---------------------------------------------------------------------------


def compute_j2j(
    pred_joints: np.ndarray, gt_joints: np.ndarray
) -> Tuple[float, np.ndarray]:
    """
    Hungarian-matched J2J.
    Returns (mean_distance, match_pred_to_gt array of length J_pred).
    """
    cost = np.linalg.norm(pred_joints[:, None, :] - gt_joints[None, :, :], axis=-1)
    row_ind, col_ind = linear_sum_assignment(cost)
    match = np.argmin(cost, axis=1)  # default: nearest GT
    match[row_ind] = col_ind  # override with optimal
    return float(cost[row_ind, col_ind].mean()), match


def compute_j2b(
    pred_joints: np.ndarray,
    gt_joints: np.ndarray,
    gt_bones: List[Tuple[int, int]],
) -> float:
    """Mean distance from each pred joint to nearest GT bone segment."""
    if not gt_bones:
        return float("nan")
    return float(
        np.mean(
            [
                min(
                    point_to_segment_distance(pj, gt_joints[p], gt_joints[c])
                    for p, c in gt_bones
                )
                for pj in pred_joints
            ]
        )
    )


def compute_b2b(
    pred_joints: np.ndarray,
    pred_bones: List[Tuple[int, int]],
    gt_joints: np.ndarray,
    gt_bones: List[Tuple[int, int]],
) -> float:
    """Symmetric Chamfer distance between pred and GT bone midpoints."""
    if not pred_bones or not gt_bones:
        return float("nan")
    pm = np.array([(pred_joints[p] + pred_joints[c]) / 2 for p, c in pred_bones])
    gm = np.array([(gt_joints[p] + gt_joints[c]) / 2 for p, c in gt_bones])
    d = np.linalg.norm(pm[:, None] - gm[None], axis=-1)
    return float((d.min(axis=1).mean() + d.min(axis=0).mean()) / 2.0)


# ---------------------------------------------------------------------------
# 5.  Skinning metrics
# ---------------------------------------------------------------------------


def project_skinning_to_vertices(
    pred_pc: np.ndarray, pred_skin: np.ndarray, gt_vertices: np.ndarray
) -> np.ndarray:
    """
    Safety projection for cases where pred pointcloud != GT vertices.
    Mirrors project_to_glb from inference.py.
    """
    out = []
    for s in range(0, len(gt_vertices), 4096):
        batch = gt_vertices[s : s + 4096]
        dist = np.linalg.norm(batch[:, None] - pred_pc[None], axis=-1)
        out.append(pred_skin[dist.argmin(axis=1)])
    return np.concatenate(out, axis=0)


def hard_assignment(weights: np.ndarray) -> np.ndarray:
    return weights.argmax(axis=1).astype(int)


def remap_pred_labels(
    pred_labels: np.ndarray, match_pred_to_gt: np.ndarray
) -> np.ndarray:
    return match_pred_to_gt[pred_labels]


def compute_skinning_metrics(
    pred_labels: np.ndarray, gt_labels: np.ndarray
) -> Dict[str, float]:
    """
    Macro-averaged per-bone IOU, Precision, Recall (binary per GT joint).
    """
    iou_l, prec_l, rec_l = [], [], []
    for j in np.unique(gt_labels):
        pj = pred_labels == j
        gj = gt_labels == j
        tp = int((pj & gj).sum())
        fp = int((pj & ~gj).sum())
        fn = int((~pj & gj).sum())
        prec_l.append(tp / (tp + fp + 1e-8))
        rec_l.append(tp / (tp + fn + 1e-8))
        iou_l.append(tp / (tp + fp + fn + 1e-8))
    return dict(
        iou=float(np.mean(iou_l)),
        precision=float(np.mean(prec_l)),
        recall=float(np.mean(rec_l)),
    )


# ---------------------------------------------------------------------------
# 6.  Per-sample evaluation
# ---------------------------------------------------------------------------


def evaluate_sample(pred: Dict, gt: Dict) -> Dict[str, float]:
    """
    Compute all metrics given a pred dict (from run_inference_on_gt_sample)
    and a gt dict (from iter_dataset_npz)
    """
    pred_joints = pred["joints"]
    gt_joints = gt["joints"]
    pred_bones = parents_to_bones(pred["parents"])
    gt_bones = adj_to_bones(gt["bones_adj"])

    # Skeleton
    j2j, match_p2g = compute_j2j(pred_joints, gt_joints)
    j2b = compute_j2b(pred_joints, gt_joints, gt_bones)
    b2b = compute_b2b(pred_joints, pred_bones, gt_joints, gt_bones)

    # Skinning: run_inference_on_gt_sample already projects to GT vertices, but guard against shape mismatch from any alternative pred source.
    pred_skin = pred["skinning_weights"]  # [V, J_pred]
    if pred_skin.shape[0] != gt["vertices"].shape[0]:
        pred_skin = project_skinning_to_vertices(
            pred["pointcloud"], pred_skin, gt["vertices"]
        )

    pred_lbl = remap_pred_labels(hard_assignment(pred_skin), match_p2g)
    gt_lbl = hard_assignment(gt["skinning_weights"])
    skin = compute_skinning_metrics(pred_lbl, gt_lbl)

    return dict(j2j=j2j, j2b=j2b, b2b=b2b, **skin)


# ---------------------------------------------------------------------------
# 7.  Model loader  (mirrors inference.py setup block)
# ---------------------------------------------------------------------------


def load_model(
    config_path: str, ckpt_path: str, device: str
) -> Tuple[torch.nn.Module, edict]:
    """Load RigARDiffusion from config + checkpoint"""
    config = edict(yaml.safe_load(open(config_path)))
    config.training.checkpoint_dir = osp.join(
        config.training.checkpoint_dir, config.training.wandb_exp_name
    )
    torch.backends.cuda.matmul.allow_tf32 = config.training.use_tf32
    torch.backends.cudnn.allow_tf32 = config.training.use_tf32

    job_overview = get_job_overview(
        num_gpus=1,
        num_epochs=config.training.num_epochs,
        num_train_samples=0,
        batch_size_per_gpu=config.training.batch_size_per_gpu,
        gradient_accumulation_steps=config.training.grad_accum_steps,
    )

    module_name, class_name = config.model.class_name.rsplit(".", 1)
    ModelClass = importlib.import_module(module_name).__dict__[class_name]
    model = ModelClass(config, device=device).to(device)

    optimizer, _, _ = configure_optimizer(
        model,
        config.training.weight_decay,
        config.training.lr,
        (config.training.beta1, config.training.beta2),
    )
    lr_scheduler = configure_lr_scheduler(
        optimizer,
        job_overview.num_param_updates,
        config.training.warmup,
        scheduler_type="cosine",
    )
    resume_job(
        ckpt_path,
        config.training.checkpoint_dir,
        model,
        optimizer,
        lr_scheduler,
        job_overview,
        config.training.warmup,
        reset_lr=False,
        reset_weight_decay=False,
        reset_training_state=False,
    )
    model.eval()
    print(f"Model loaded from {ckpt_path}")
    return model, config


# ---------------------------------------------------------------------------
# 8.  Dataset-level evaluation loop
# ---------------------------------------------------------------------------


def evaluate_dataset(
    config_path: str,
    ckpt_path: str,
    dataset_npz: str,
    metadata_csv: Optional[str] = None,
    output_csv: str = "results.csv",
    category_filter: Optional[str] = None,
    device: str = "cuda:0",
    amp_dtype: str = "bf16",
    max_meshes: Optional[int] = None,
    use_wandb: bool = False,
    wandb_project: str = "riganything-eval",
    wandb_name: Optional[str] = None,
) -> pd.DataFrame:
    """
    Load the model once, iterate the single dataset NPZ, run inference on
    each GT mesh, and evaluate all metrics
    """
    torch.cuda.set_device(device)
    model, config = load_model(config_path, ckpt_path, device)

    amp_map = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}
    use_amp = config.training.use_amp
    amp_type = amp_map.get(amp_dtype, torch.bfloat16)

    records, skipped = [], 0

    # Initialize wandb if requested
    if use_wandb:
        wandb.init(
            project=wandb_project,
            name=wandb_name,
            config={
                "config_path": config_path,
                "ckpt_path": ckpt_path,
                "dataset_npz": dataset_npz,
                "category_filter": category_filter,
                "max_meshes": max_meshes,
                "amp_dtype": amp_dtype,
                "model_config": dict(config),
            },
        )

    with torch.no_grad(), torch.autocast(
        enabled=use_amp, device_type="cuda", dtype=amp_type
    ):
        for step, gt in enumerate(
            tqdm(
                iter_dataset_npz(dataset_npz, metadata_csv, category_filter),
                desc="Evaluating",
            )
        ):
            if max_meshes is not None and len(records) >= max_meshes:
                print(f"\nReached max_meshes={max_meshes}, stopping early.")
                break

            uuid = gt["uuid"]
            try:
                pred = run_inference_on_gt_sample(gt, model, device)
                metrics = evaluate_sample(pred, gt)
            except Exception as exc:
                print(f"[ERROR] {uuid}: {exc}")
                skipped += 1
                continue

            meta = gt.get("meta", {})

            # 1. Create the record with expanded metadata (including mesh size)
            record = dict(
                uuid=uuid,
                category_label=meta.get("category_label", "Unknown"),
                joint_count=meta.get("joint_count", -1),
                bone_count=meta.get("bone_count", -1),
                pred_joints=len(pred["joints"]),
                vertex_count=len(gt["vertices"]),
                face_count=len(gt["faces"]),
                **metrics,
            )
            records.append(record)

            # 2. Live update to wandb dashboard including the metadata
            if use_wandb:
                wandb.log(
                    {
                        # Core Metrics
                        "eval/j2j": metrics["j2j"],
                        "eval/j2b": metrics["j2b"],
                        "eval/b2b": metrics["b2b"],
                        "eval/iou": metrics["iou"],
                        "eval/precision": metrics["precision"],
                        "eval/recall": metrics["recall"],
                        # Metadata & Mesh Size Characteristics
                        "meta/category": record["category_label"],
                        "meta/gt_joint_count": record["joint_count"],
                        "meta/gt_bone_count": record["bone_count"],
                        "meta/pred_joint_count": record["pred_joints"],
                        "meta/vertex_count": record["vertex_count"],
                        "meta/face_count": record["face_count"],
                        "step": step,
                    }
                )

    df = pd.DataFrame(records)

    if len(df) > 0:
        metric_cols = ["j2j", "j2b", "b2b", "iou", "precision", "recall"]
        print("\n===== Aggregate Results =====")
        for col in metric_cols:
            print(f"  {col:<12}: {df[col].mean():.4f}")
        print(f"\n  Evaluated : {len(df)}")
        print(f"  Skipped   : {skipped}")

        df.to_csv(output_csv, index=False)
        print(f"\nPer-sample results saved to: {output_csv}")

        if df["category_label"].nunique() > 1:
            print("\n===== Per-Category Results =====")
            print(df.groupby("category_label")[metric_cols].mean().to_string())

        # Log summary metrics and the full dataframe as a table to wandb
        if use_wandb:
            for col in metric_cols:
                wandb.summary[f"mean_{col}"] = df[col].mean()
            wandb.log({"results_table": wandb.Table(dataframe=df)})
    else:
        print("No samples were successfully evaluated.")

    if use_wandb:
        wandb.finish()

    torch.cuda.empty_cache()
    return df


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate RigAnything on Articulation-XL2.0 (single NPZ)"
    )
    parser.add_argument("--config", required=True, help="Path to config.yaml")
    parser.add_argument(
        "--ckpt", required=True, help="Path to RigAnything checkpoint (.pt)"
    )
    parser.add_argument(
        "--dataset_npz", required=True, help="Path to articulation_xlv2_test.npz"
    )
    parser.add_argument(
        "--metadata_csv",
        default=None,
        help="Optional metadata CSV (for category labels)",
    )
    parser.add_argument("--output_csv", default="results.csv")
    parser.add_argument(
        "--category", default=None, help="Restrict to one category_label"
    )
    parser.add_argument(
        "--max_meshes",
        default=None,
        type=int,
        help="Stop after this many successfully evaluated meshes",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--amp_dtype", default="bf16", choices=["fp16", "bf16", "fp32"])

    # WandB arguments
    parser.add_argument(
        "--use_wandb", action="store_true", help="Enable logging to Weights & Biases"
    )
    parser.add_argument(
        "--wandb_project", default="riganything-eval", help="WandB project name"
    )
    parser.add_argument(
        "--wandb_name", default=None, help="Optional specific name for this WandB run"
    )

    args = parser.parse_args()

    evaluate_dataset(
        config_path=args.config,
        ckpt_path=args.ckpt,
        dataset_npz=args.dataset_npz,
        metadata_csv=args.metadata_csv,
        output_csv=args.output_csv,
        category_filter=args.category,
        device=args.device,
        amp_dtype=args.amp_dtype,
        max_meshes=args.max_meshes,
        use_wandb=args.use_wandb,
        wandb_project=args.wandb_project,
        wandb_name=args.wandb_name,
    )
