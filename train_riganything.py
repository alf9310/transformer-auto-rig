"""
Fine-tuning of RigAnything on a specific 3D model category.

Reuses from evaluate_riganything.py:
  - iter_dataset_npz / _unpack_arr0 / inspect_dataset_npz  (dataset loading)
  - load_model                                              (model + checkpoint)
  - _sample_surface_points                                  (point cloud sampling)
  - smooth_weights_per_vertex / project_to_glb              (post-processing)

Reuses from the existing training infrastructure (inference.py / utils/):
  - configure_optimizer / configure_lr_scheduler
  - checkpoint_job / resume_job / get_job_overview

The training step mirrors the model's own forward design:
  1. Encode the 1024-pt point cloud -> pc_tokens
  2. Autoregressively process GT joint tokens through the transformer
  3. Diffusion loss on joint positions           (diffloss.forward)
  4. Cross-entropy loss on parent prediction     (parents_decoder)
  5. MSE loss on skinning weights                (skinning_mlp)

Usage:
    python train_riganything.py \\
        --config       config.yaml                \\
        --ckpt         ckpt/riganything_ckpt.pt   \\
        --dataset_npz  data/articulation_xlv2_test.npz \\
        --metadata_csv data/meta_Articulation_XL_2.0.csv \\
        --category     humanoid                   \\
        --output_dir   ckpt/finetune_humanoid      \\
        [--max_steps   5000]                       \\
        [--lr          1e-5]                       \\
        [--use_wandb]
"""

from __future__ import annotations

import argparse
import os
import os.path as osp
import random
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import trimesh
import wandb
from torch.cuda.amp import GradScaler
from tqdm import tqdm

from evaluate_riganything import (
    _N_POINTS,
    _SMOOTH_FACTOR,
    _SMOOTH_ITERS,
    _SKIN_THRESH,
    _TOPK,
    _sample_surface_points,
    _unpack_arr0,
    inspect_dataset_npz,
    iter_dataset_npz,
    load_model,
    project_to_glb,
    smooth_weights_per_vertex,
)

# Reuse checkpoint utilities directly
from utils.job_checkpointer import checkpoint_job, get_job_overview, resume_job
from utils.optimizer_scheduler import configure_lr_scheduler, configure_optimizer

# ---------------------------------------------------------------------------
# 1.  Dataset: collate GT samples into model-ready tensors
# ---------------------------------------------------------------------------


def collate_sample(gt: Dict, device: str) -> Optional[Dict[str, torch.Tensor]]:
    """
    Convert one raw GT dict (from iter_dataset_npz) into the batch dict that
    RigARDiffusion's transformer expects.

    Same structure as inference.py's batch dict, but with ground-truth joints/parents/skinning populated.

    Returns None if the sample cannot be used for training (e.g. too few joints).
    """
    vertices = gt["vertices"]  # [V, 3]
    faces = gt["faces"]  # [F, 3]
    normals = gt["normals"]  # [V, 3]
    joints = gt["joints"]  # [J_gt, 3]
    bones_adj = gt["bones_adj"]  # [J_gt, J_gt]
    skin_gt = gt["skinning_weights"]  # [V, J_gt]

    n_joints_gt = joints.shape[0]
    max_joints = 64  # model cap, matches config.model.joints_tokenizer.n_joints

    if n_joints_gt < 2:
        return None  # skip degenerate skeletons
    if n_joints_gt > max_joints:
        joints = joints[:max_joints]
        bones_adj = bones_adj[:max_joints, :max_joints]
        skin_gt = skin_gt[:, :max_joints]
        n_joints_gt = max_joints

    # ---- 1024-point surface sample -----------------------------------------
    pc_w_norm = gt.get("pc_w_norm")
    if pc_w_norm is not None and pc_w_norm.shape[0] == _N_POINTS:
        pts = pc_w_norm[:, :3].astype(np.float32)
        norms = pc_w_norm[:, 3:].astype(np.float32)
    else:
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
        pts, norms = _sample_surface_points(mesh, _N_POINTS)

    # ---- Normalise to unit sphere (mirrors inference.py) -------------------
    center = (np.max(pts, axis=0) + np.min(pts, axis=0)) / 2.0
    scale = np.max(np.abs(pts - center)) + 1e-8
    pts_norm = (pts - center) / scale
    joints_norm = (joints - center) / scale

    # ---- Build parent index array from adjacency matrix --------------------
    # bones_adj[i,j]=1 means bone from joint i to joint j (i is parent of j).
    # For each joint find its parent; root is its own parent.
    parents = np.arange(n_joints_gt, dtype=np.int32)  # default: self-loop = root
    rows, cols = np.where(bones_adj[:n_joints_gt, :n_joints_gt])
    for r, c in zip(rows, cols):
        parents[c] = r  # r is parent of c

    # ---- Pad joints / parents / skinning to max_joints ---------------------
    joints_pad = np.zeros((max_joints, 3), dtype=np.float32)
    parents_pad = np.arange(max_joints, dtype=np.int32)  # self-loop padding
    skin_pad = np.zeros((_N_POINTS, max_joints), dtype=np.float32)

    joints_pad[:n_joints_gt] = joints_norm
    parents_pad[:n_joints_gt] = parents

    # Project GT vertex skinning -> sampled point cloud skinning
    # (inverse of project_to_glb; use nearest-vertex lookup)
    dist_pc_to_verts = np.linalg.norm(
        pts[:, None, :] - vertices[None, :, :], axis=-1
    )  # [1024, V]
    nn_vert_idx = dist_pc_to_verts.argmin(axis=1)  # [1024]
    skin_on_pc = skin_gt[nn_vert_idx]  # [1024, J_gt]
    # Normalise (GT weights should already sum to 1, but guard anyway)
    skin_on_pc = skin_on_pc / (skin_on_pc.sum(axis=1, keepdims=True) + 1e-8)
    skin_pad[:, :n_joints_gt] = skin_on_pc

    # ---- Joint existence mask [max_joints] ---------------------------------
    joint_mask = np.zeros(max_joints, dtype=np.float32)
    joint_mask[:n_joints_gt] = 1.0

    return {
        "pointcloud": torch.from_numpy(pts_norm)
        .unsqueeze(0)
        .to(device),  # [1, 1024, 3]
        "normals": torch.from_numpy(norms).unsqueeze(0).to(device),  # [1, 1024, 3]
        "joints": torch.from_numpy(joints_pad).unsqueeze(0).to(device),  # [1, J, 3]
        "parents": torch.from_numpy(parents_pad.astype(np.int64))
        .unsqueeze(0)
        .to(device),
        "skinning_weights": torch.from_numpy(skin_pad)
        .unsqueeze(0)
        .to(device),  # [1, 1024, J]
        "joint_mask": torch.from_numpy(joint_mask).unsqueeze(0).to(device),  # [1, J]
        "scale": torch.tensor([scale]).to(device),
        "center": torch.from_numpy(center).to(device),
        "item_idx": [gt["uuid"]],
        "root_idx": torch.tensor([int(gt["root_index"])]).to(device),
        "n_joints": n_joints_gt,
    }


# ---------------------------------------------------------------------------
# 2.  Training step
# ---------------------------------------------------------------------------


def training_step(
    batch: Dict,
    model: nn.Module,
    config,
) -> Dict[str, torch.Tensor]:
    """
    One forward pass for fine-tuning.

    The model's autoregressive design (RigARDiffusion) has three learnable components:

    (a) Joint position loss: via model.diffloss (DiffLoss / diffusion head).
        DiffLoss.forward(target, z) expects:
          target : [B*J, 3]  ground-truth joint XYZ (normalised)
          z      : [B*J, d]  per-joint conditioning token from the transformer

    (b) Parent prediction: loss cross-entropy on the parents_decoder output.
        For each joint j, we predict which of the preceding joints is its parent.

    (c) Skinning weight loss: MSE between skinning_mlp output and GT weights
        on the sampled point cloud.

    All three losses are gated by joint_mask so padding slots don't contribute.
    """
    B = batch["pointcloud"].shape[0]  # always 1 for per-sample fine-tuning
    n_joints = batch["n_joints"]
    device = batch["pointcloud"].device
    max_joints = batch["joints"].shape[1]  # 64

    # Encode point cloud
    pointcloud_input = torch.cat(
        [batch["pointcloud"], batch["normals"]], dim=-1
    )  # [B, 1024, 6]
    pc_tokens = model.pc_tokenizer(pointcloud_input)  # [B, 1024, d]

    # Tokenize GT joints
    gt_joints_norm = batch["joints"]  # [B, J, 3]
    gt_parents = batch["parents"]  # [B, J]
    joint_mask = batch["joint_mask"]  # [B, J]  1=real, 0=pad

    # Joint position tokens [B, J, d]
    joint_pos_tokens = model.joint_tokenizer(gt_joints_norm)

    # Parent position tokens [B, J, d] (each joint gets its parent's pos token)
    parent_idx_expanded = gt_parents.unsqueeze(-1).expand(
        -1, -1, joint_pos_tokens.shape[-1]
    )
    parent_pos_tokens = torch.gather(joint_pos_tokens, 1, parent_idx_expanded)

    # Positional embeddings [B, J, d]
    joint_idx_emb = (
        model.joint_index_pos_embedding.unsqueeze(0)
        .expand(B, -1, -1)
        .to(device)[:, :max_joints, :]
    )
    parent_idx_emb = torch.gather(joint_idx_emb, 1, parent_idx_expanded)

    # Fused joint input tokens [B, J, d] for the transformer sequence
    # Concatenate: joint_pos_emb, joint_pos_token, parent_pos_emb, parent_pos_token
    joint_input = torch.cat(
        [joint_idx_emb, joint_pos_tokens, parent_idx_emb, parent_pos_tokens], dim=-1
    )  # [B, J, 4d]
    joint_tokens_ar = model.joint_mlp(joint_input)  # [B, J, d]

    # Start token prepended to joint sequence -> [B, J+1, d]
    start_token = model.start_token.expand(B, 1, -1)
    # The transformer sees: [pc_tokens | start | joint_0 | ... | joint_{J-1}]
    # During training the full sequence is provided (teacher forcing)
    input_tokens = torch.cat(
        [pc_tokens, start_token, joint_tokens_ar], dim=1
    )  # [B, 1024+1+J, d]

    # Custom causal attention mask over the full training sequence
    n_pc = pc_tokens.shape[1]
    attn_mask = model.get_custom_attn_mask(n_pc, max_joints + 1).to(device)
    # Trim to actual sequence length
    seq_len = input_tokens.shape[1]
    attn_mask = attn_mask[:seq_len, :seq_len]

    # Transformer forward (all layers, with gradient checkpointing)
    # Zero KV caches: in-place cache writes are side effects that corrupt
    # gradient checkpointing recomputation. Clear before every forward pass.
    for layer in model.transformer:
        layer.attn.k_cache.zero_()
        layer.attn.v_cache.zero_()

    input_tokens = model.transformer_input_layernorm(input_tokens)

    # Direct layer iteration, no checkpointing avoids recomputed cache writes
    for layer in model.transformer:
        input_tokens = layer(input_tokens, attn_mask=attn_mask, start_pos=0)

    # Joint conditioning tokens are the positions after the start token
    # token at position n_pc+1+j is the output for joint j
    joint_out_tokens = input_tokens[:, n_pc + 1 :, :]  # [B, J, d]

    # -- (a) Diffusion loss on joint positions -------------------------------
    # Only on real (non-padding) joints; gate with mask
    mask_flat = joint_mask.reshape(-1)  # [B*J]
    targets_flat = gt_joints_norm.reshape(-1, 3)  # [B*J, 3]
    z_flat = joint_out_tokens.reshape(-1, joint_out_tokens.shape[-1])  # [B*J, d]

    diff_loss = model.diffloss(targets_flat, z_flat, mask=mask_flat)

    # -- (b) Parent prediction loss (cross-entropy) --------------------------
    # Fuse joint output tokens with current position / index embeddings
    joint_fuse_input = torch.cat(
        [joint_out_tokens, joint_idx_emb, joint_pos_tokens], dim=-1
    )  # [B, J, 3d]
    joints_fused = model.joint_fuse_mlp(joint_fuse_input)  # [B, J, d]

    # Pairwise concat for parent prediction [B, J, J, 2d]
    parent_candidates = model.concat_parent_candidate_features(joints_fused)
    parent_logits = model.parents_decoder(parent_candidates).squeeze(-1)  # [B, J, J]

    # GT parent labels: for padding joints use self-index (ignored by mask)
    gt_parent_labels = gt_parents.clone()  # [B, J]

    # Mask: only compute loss for real joints (not root, not padding)
    is_non_root = gt_parents != torch.arange(max_joints, device=device).unsqueeze(0)
    real_mask = joint_mask.bool() & is_non_root  # [B, J]

    parent_loss = torch.tensor(0.0, device=device)
    if real_mask.any():
        logits_sel = parent_logits[real_mask]  # [N_real, J]
        labels_sel = gt_parent_labels[real_mask]  # [N_real]
        parent_loss = F.cross_entropy(logits_sel, labels_sel)

    # -- (c) Skinning weight loss (MSE) --------------------------------------
    # Concat pc_tokens [B, 1024, d] with joint tokens [B, J, d]
    pc_tokens_out = input_tokens[:, :n_pc, :]  # [B, 1024, d]
    skinning_tokens = model.concat_pc_joint_features(
        pc_tokens_out, joint_out_tokens
    )  # [B, J, 1024, 2d]
    skin_pred = model.skinning_mlp(skinning_tokens).view(
        B, max_joints, -1
    )  # [B, J, 1024]
    # Transpose to [B, 1024, J] to match GT
    skin_pred = skin_pred.permute(0, 2, 1)  # [B, 1024, J]
    skin_pred = F.softmax(skin_pred, dim=-1)

    skin_gt = batch["skinning_weights"]  # [B, 1024, J]
    # Only supervise real joint channels
    joint_mask_skin = joint_mask.unsqueeze(1).expand_as(skin_pred)  # [B, 1024, J]
    skin_loss = F.mse_loss(skin_pred * joint_mask_skin, skin_gt * joint_mask_skin)

    # -- Combine losses with configurable weights ----------------------------
    w_diff = config.training.get("diff_loss_weight", 1.0)
    w_parent = config.training.get("parent_loss_weight", 1.0)
    w_skin = config.training.get("skin_loss_weight", 0.5)

    total_loss = w_diff * diff_loss + w_parent * parent_loss + w_skin * skin_loss

    return {
        "loss": total_loss,
        "diff_loss": diff_loss,
        "parent_loss": parent_loss,
        "skin_loss": skin_loss,
    }


# ---------------------------------------------------------------------------
# 3.  Dataset wrapper: infinite shuffled stream from iter_dataset_npz
# ---------------------------------------------------------------------------


def infinite_stream(
    npz_path: str,
    metadata_csv: Optional[str],
    category_filter: Optional[str],
    device: str,
    seed: int = 42,
):
    """
    Yields collated training batches forever, reshuffling on each epoch.
    Skips samples that collate_sample cannot use (too few joints, etc.).
    """
    rng = random.Random(seed)
    epoch = 0
    while True:
        # Collect all valid samples for this epoch into memory-light index list
        samples = list(iter_dataset_npz(npz_path, metadata_csv, category_filter))
        rng.shuffle(samples)
        epoch += 1
        n_valid = 0
        for gt in samples:
            batch = collate_sample(gt, device)
            if batch is None:
                continue
            n_valid += 1
            yield batch
        print(f"[Epoch {epoch}] {n_valid}/{len(samples)} samples used")


# ---------------------------------------------------------------------------
# 4.  Main fine-tuning loop
# ---------------------------------------------------------------------------


def finetune(
    config_path: str,
    ckpt_path: str,
    dataset_npz: str,
    output_dir: str,
    metadata_csv: Optional[str] = None,
    category_filter: Optional[str] = None,
    max_steps: int = 5000,
    lr: float = 1e-5,
    grad_accum_steps: int = 4,
    save_every: int = 500,
    device: str = "cuda:0",
    amp_dtype: str = "bf16",
    seed: int = 42,
    use_wandb: bool = False,
    wandb_project: str = "riganything-finetune",
    wandb_name: Optional[str] = None,
) -> None:
    """
    Fine-tune a pre-trained RigAnything checkpoint on a specific object category.

    Args:
        config_path     : Path to config.yaml (same one used for pre-training)
        ckpt_path       : Pre-trained checkpoint to start from
        dataset_npz     : Packed Articulation-XL2.0 NPZ
        output_dir      : Directory to save fine-tuned checkpoints
        metadata_csv    : Optional metadata CSV for category labels / filtering
        category_filter : If set, only train on this category_label
        max_steps       : Total parameter-update steps
        lr              : Peak learning rate for fine-tuning (typically << pre-train LR)
        grad_accum_steps: Gradient accumulation before each optimizer step
        save_every      : Save a checkpoint every N param-update steps
        device          : CUDA device string
        amp_dtype       : Mixed-precision dtype ('fp16', 'bf16', 'fp32')
        seed            : Random seed for dataset shuffling
        use_wandb       : Log to Weights & Biases
        wandb_project   : W&B project name
        wandb_name      : Optional W&B run name
    """
    torch.cuda.set_device(device)
    torch.manual_seed(seed)
    np.random.seed(seed)
    os.makedirs(output_dir, exist_ok=True)

    # -- Load pretrained model (reuse load_model from evaluate_riganything) --
    model, config = load_model(config_path, ckpt_path, device)
    model.train()

    # Override LR and grad_accum from CLI args (fine-tune typically uses smaller LR)
    config.training.lr = lr
    config.training.grad_accum_steps = grad_accum_steps

    # -- Optimizer and scheduler (reuse existing utilities) ------------------
    optimizer, optim_param_dict, _ = configure_optimizer(
        model,
        config.training.weight_decay,
        lr,
        (config.training.beta1, config.training.beta2),
    )
    lr_scheduler = configure_lr_scheduler(
        optimizer,
        total_train_steps=max_steps,
        warm_up_steps=min(config.training.warmup, max_steps // 10),
        scheduler_type="cosine",
    )

    # -- AMP -----------------------------------------------------------------
    amp_map = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}
    use_amp = amp_dtype != "fp32"
    amp_type = amp_map[amp_dtype]
    enable_scaler = use_amp and amp_dtype == "fp16"
    scaler = GradScaler("cuda", enabled=enable_scaler)

    # -- W&B -----------------------------------------------------------------
    if use_wandb:
        wandb.init(
            project=wandb_project,
            name=wandb_name or f"finetune_{category_filter or 'all'}",
            config={
                "config_path": config_path,
                "ckpt_path": ckpt_path,
                "category_filter": category_filter,
                "max_steps": max_steps,
                "lr": lr,
                "grad_accum_steps": grad_accum_steps,
                "amp_dtype": amp_dtype,
                "seed": seed,
            },
        )

    # -- Data stream ---------------------------------------------------------
    stream = infinite_stream(
        dataset_npz, metadata_csv, category_filter, device, seed=seed
    )

    # -- Training loop -------------------------------------------------------
    param_update_step = 0
    fwdbwd_step = 0
    optimizer.zero_grad()

    pbar = tqdm(total=max_steps, desc="Fine-tuning")

    while param_update_step < max_steps:
        batch = next(stream)

        # Forward + loss
        with torch.autocast(enabled=use_amp, device_type="cuda", dtype=amp_type):
            losses = training_step(batch, model, config)
            loss = losses["loss"] / grad_accum_steps

        # Backward
        scaler.scale(loss).backward()
        fwdbwd_step += 1

        # Gradient accumulation
        if fwdbwd_step % grad_accum_steps == 0:
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            lr_scheduler.step()
            optimizer.zero_grad()
            param_update_step += 1

            # -- Logging -----------------------------------------------------
            log_dict = {
                "train/loss": losses["loss"].item(),
                "train/diff_loss": losses["diff_loss"].item(),
                "train/parent_loss": losses["parent_loss"].item(),
                "train/skin_loss": losses["skin_loss"].item(),
                "train/lr": optimizer.param_groups[0]["lr"],
                "step": param_update_step,
            }
            if use_wandb:
                wandb.log(log_dict)

            pbar.set_postfix(
                {
                    "loss": f"{losses['loss'].item():.4f}",
                    "diff": f"{losses['diff_loss'].item():.4f}",
                    "parent": f"{losses['parent_loss'].item():.4f}",
                    "skin": f"{losses['skin_loss'].item():.4f}",
                    "lr": f"{optimizer.param_groups[0]['lr']:.2e}",
                }
            )
            pbar.update(1)

            # -- Checkpoint --------------------------------------------------
            if param_update_step % save_every == 0 or param_update_step >= max_steps:
                checkpoint_job(
                    out_dir=output_dir,
                    model=model,
                    optimizer=optimizer,
                    lr_scheduler=lr_scheduler,
                    fwdbwd_pass_step=fwdbwd_step,
                    param_update_step=param_update_step,
                )
                print(f"\nCheckpoint saved at step {param_update_step}")

    pbar.close()
    print(f"\nFine-tuning complete. Checkpoints saved to: {output_dir}")

    if use_wandb:
        wandb.finish()


# ---------------------------------------------------------------------------
# 5.  CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fine-tune RigAnything on a specific 3D object category"
    )

    # -- Required ------------------------------------------------------------
    parser.add_argument("--config", required=True, help="Path to config.yaml")
    parser.add_argument(
        "--ckpt", required=True, help="Pre-trained checkpoint (.pt) to start from"
    )
    parser.add_argument(
        "--dataset_npz",
        required=True,
        help="Path to articulation_xlv2_test.npz (or train split)",
    )
    parser.add_argument(
        "--output_dir", required=True, help="Directory to save fine-tuned checkpoints"
    )

    # -- Dataset -------------------------------------------------------------
    parser.add_argument(
        "--metadata_csv", default=None, help="Optional metadata CSV for category labels"
    )
    parser.add_argument(
        "--category",
        default=None,
        help="category_label to fine-tune on (e.g. 'humanoid'). "
        "Omit to fine-tune on the full dataset.",
    )

    # -- Training hyperparams ------------------------------------------------
    parser.add_argument(
        "--max_steps", default=5000, type=int, help="Total number of optimizer steps"
    )
    parser.add_argument(
        "--lr",
        default=1e-5,
        type=float,
        help="Peak learning rate (recommend << pre-training LR)",
    )
    parser.add_argument(
        "--grad_accum",
        default=4,
        type=int,
        help="Gradient accumulation steps (effective batch size)",
    )
    parser.add_argument(
        "--save_every",
        default=500,
        type=int,
        help="Save a checkpoint every N optimizer steps",
    )
    parser.add_argument("--seed", default=42, type=int)

    # -- Hardware ------------------------------------------------------------
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--amp_dtype", default="bf16", choices=["fp16", "bf16", "fp32"])

    # -- W&B -----------------------------------------------------------------
    parser.add_argument(
        "--use_wandb", action="store_true", help="Enable Weights & Biases logging"
    )
    parser.add_argument(
        "--wandb_project", default="riganything-finetune", help="W&B project name"
    )
    parser.add_argument("--wandb_name", default=None, help="Optional W&B run name")

    args = parser.parse_args()

    finetune(
        config_path=args.config,
        ckpt_path=args.ckpt,
        dataset_npz=args.dataset_npz,
        output_dir=args.output_dir,
        metadata_csv=args.metadata_csv,
        category_filter=args.category,
        max_steps=args.max_steps,
        lr=args.lr,
        grad_accum_steps=args.grad_accum,
        save_every=args.save_every,
        device=args.device,
        amp_dtype=args.amp_dtype,
        seed=args.seed,
        use_wandb=args.use_wandb,
        wandb_project=args.wandb_project,
        wandb_name=args.wandb_name,
    )
