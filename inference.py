import argparse
import copy
import importlib
import os
import os.path as osp
from copy import deepcopy

import bpy
import numpy as np

import torch
import torch.nn.functional as F
import trimesh
import yaml
from easydict import EasyDict as edict
from utils.job_checkpointer import resume_job, get_job_overview
from utils.optimizer_scheduler import configure_lr_scheduler, configure_optimizer
from tqdm import tqdm


def smooth_weights_per_vertex(mesh, weights, iterations=5, neighbor_factor=0.3):
    """
    Smooth weights using neighbor averaging

    Parameters:
    mesh: trimesh.Trimesh object
    weights: numpy array of shape (num_vertices, num_bones)
    iterations: number of smoothing iterations
    neighbor_factor: influence of neighbors (0-1)

    Returns:
    smoothed weights
    """
    smoothed_weights = deepcopy(weights)

    # Get vertex neighbors of bpy mesh
    vertex_neighbors = mesh.vertex_neighbors

    for _ in range(iterations):
        new_weights = deepcopy(smoothed_weights)

        for i in range(len(mesh.vertices)):
            # Get neighbors for this vertex
            neighbors = vertex_neighbors[i]

            if len(neighbors) > 0:
                # Get average weight of neighbors
                neighbor_weights = np.mean(smoothed_weights[neighbors], axis=0)

                # Blend with original weights
                new_weights[i] = (1.0 - neighbor_factor) * smoothed_weights[
                    i
                ] + neighbor_factor * neighbor_weights

                # Ensure weights sum to 1
                if np.sum(new_weights[i]) > 0:
                    new_weights[i] /= np.sum(new_weights[i])

        smoothed_weights = new_weights

    return smoothed_weights


def project_to_glb(glb_vert, mesh_vert, mesh_skinning):
    # For each glb vertex, find the closest mesh vertex
    # Then assign the mesh skinning weights to the glb vertex
    # This is a naive implementation, can be optimized
    glb_skinning = []
    glb_vert = torch.from_numpy(glb_vert)
    mesh_vert = torch.from_numpy(mesh_vert).to(device)
    mesh_skinning = torch.from_numpy(mesh_skinning).to(device)

    from tqdm import tqdm

    # batchly process
    for glb_vert_bs in tqdm(torch.split(glb_vert, 4096, dim=0)):
        glb_vert_bs = glb_vert_bs.to(mesh_vert.device)
        dist = torch.norm(glb_vert_bs[:, None] - mesh_vert, dim=2)
        closest_idx = torch.argmin(dist, dim=1)
        glb_skinning_bs = mesh_skinning[closest_idx]
        glb_skinning_bs = glb_skinning_bs.detach().cpu().numpy()
        glb_skinning.append(glb_skinning_bs)
        # glb_skinning = glb_skinning.detach().cpu().numpy()
    glb_skinning = np.concatenate(glb_skinning, axis=0)

    return glb_skinning


device = "cuda:0"  # Use first GPU
torch.cuda.set_device(device)
torch.manual_seed(777)  # Single consistent seed
print(f"Using device {device}")
ddp_rank = 0

##############################################################################################################
# Config setup
##############################################################################################################
def set_nested_key(data, keys, value):
    """Sets value in nested dictionary"""
    key = keys.pop(0)

    if keys:
        if key not in data:
            data[key] = {}
        set_nested_key(data[key], keys, value)
    else:
        data[key] = value_type(value)


def value_type(value):
    """Convert str to bool/int/float if possible"""
    try:
        if value.lower() == "true":
            return True
        elif value.lower() == "false":
            return False
        else:
            try:
                return int(value)
            except ValueError:
                try:
                    return float(value)
                except ValueError:
                    return value
    except AttributeError:
        return value


parser = argparse.ArgumentParser(description="Override YAML values")
parser.add_argument(
    "--config", "-c", type=str, required=True, help="Path to YAML configuration file"
)
parser.add_argument(
    "--load", type=str, default="", help="Force to load the weight from somewhere else"
)
parser.add_argument(
    "--set",
    "-s",
    type=str,
    action="append",
    nargs=2,
    metavar=("KEY", "VALUE"),
    help="New value for the key",
)
parser.add_argument("--mesh_path", type=str, default="", help="Path to mesh file")
args = parser.parse_args()

config = yaml.safe_load(open(args.config, "r"))

# Override the YAML values
if args.set is not None:
    for key_value in args.set:
        key_parts = key_value[0].split(".")
        value = key_value[1]
        set_nested_key(config, key_parts, value)

config = edict(config)
print(config)

##############################################################################################################
# Update the output path using the given exp name
##############################################################################################################
config.training.checkpoint_dir = osp.join(
    config.training.checkpoint_dir, config.training.wandb_exp_name
)
config.training.checkpoint_dir_s3 = osp.join(
    config.training.checkpoint_dir_s3, config.training.wandb_exp_name
)
# config.inference_out_dir = osp.join(
#     config.inference_out_dir, config.training.wandb_exp_name
# )
# if config.get("evaluation", False):
#     config.evaluation_out_dir = osp.join(
#         config.evaluation_out_dir, config.training.wandb_exp_name
#     )

##############################################################################################################
# tf32 setup
##############################################################################################################
torch.backends.cuda.matmul.allow_tf32 = config.training.use_tf32
torch.backends.cudnn.allow_tf32 = config.training.use_tf32

##############################################################################################################
# Overview job
##############################################################################################################
job_overview = get_job_overview(
    num_gpus=1,
    num_epochs=config.training.num_epochs,
    num_train_samples=0,
    batch_size_per_gpu=config.training.batch_size_per_gpu,
    gradient_accumulation_steps=config.training.grad_accum_steps,
    max_fwdbwd_passes=config.training.get("max_fwdbwd_passes", int(1e10)),
)
print(job_overview)

##############################################################################################################
# model setup; dynamic import
##############################################################################################################
module, class_name = config.model.class_name.rsplit(".", 1)
Gaussians2Rig = importlib.import_module(module).__dict__[class_name]
model = Gaussians2Rig(config, device=device).to(device)
model_overview = model.get_overview()

##############################################################################################################
# Load checkpoint
##############################################################################################################
optimizer, optim_param_dict, all_param_dict = configure_optimizer(
    model,
    config.training.weight_decay,
    config.training.lr,
    (config.training.beta1, config.training.beta2),
)
optim_param_list = list(optim_param_dict.values())
optimizer_overview = edict(
    num_optim_params=sum(p.numel() for n, p in optim_param_dict.items()),
    num_all_params=sum(p.numel() for n, p in all_param_dict.items()),
    optim_param_names=list(optim_param_dict.keys()),
    freeze_param_names=list(set(all_param_dict.keys()) - set(optim_param_dict.keys())),
)
if ddp_rank == 0:
    print(optimizer_overview)

lr_scheduler = configure_lr_scheduler(
    optimizer,
    job_overview.num_param_updates,
    config.training.warmup,
    scheduler_type="cosine",
)
fwdbwd_pass_step, param_update_step = 0, 0

for try_load_path in [
    args.load,
    config.training.checkpoint_dir,
    config.training.get("resume_ckpt", ""),
]:
    if config.training.get("force_resume_ckpt", False):
        try_load_path = config.training.resume_ckpt

    reset_training_state = config.training.get("reset_training_state", False) and (
        try_load_path == config.training.get("resume_ckpt", "")
    )  # only respect reset_training_state if it's the resume_ckpt
    optimizer, lr_scheduler, fwdbwd_pass_step, param_update_step = resume_job(
        try_load_path,
        config.training.checkpoint_dir,
        model,
        optimizer,
        lr_scheduler,
        job_overview,
        config.training.warmup,
        config.training.reset_lr,
        config.training.reset_weight_decay,
        reset_training_state,
    )

    if fwdbwd_pass_step > 0:
        break

##############################################################################################################
# AMP setup
##############################################################################################################
enable_grad_scaler = config.training.use_amp and config.training.amp_dtype == "fp16"
scaler = torch.cuda.amp.GradScaler(enabled=enable_grad_scaler)
amp_dtype_mapping = {
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
    "fp32": torch.float32,
}
print(f"Grad scaler enabled: {enable_grad_scaler}")

##############################################################################################################
# Inference loop
##############################################################################################################
if config.inference or config.get("evaluation", False):
    if config.inference:
        print(f"Running inference; save results to: {config.inference_out_dir}")
    else:
        print(f"Running evaluation; save results to: {config.evaluation_out_dir}")

    model.eval()
    with torch.no_grad(), torch.autocast(
        enabled=config.training.use_amp,
        device_type="cuda",
        dtype=amp_dtype_mapping[config.training.amp_dtype],
    ):
        item_list = [args.mesh_path]
        for idx, file_path in tqdm(enumerate(item_list)):
            # Load from obj file
            item_idx = osp.basename(file_path).split(".")[0]
            batch_dir = config.inference_out_dir
            # batch_dir = osp.join(config.inference_out_dir, item_idx)

            # Load using bpy
            bpy.ops.wm.read_factory_settings(use_empty=True)
            # Clear existing scene
            bpy.ops.object.select_all(action="SELECT")
            bpy.ops.object.delete()
            bpy.ops.import_scene.gltf(filepath=file_path)
            # Initialize arrays to store all vertices in glb (may have duplicates)
            full_points_glb = []
            # Traverse all mesh objects in the scene and sorted
            mesh_object_list = []
            trimesh_list = []
            for obj in bpy.data.objects:
                if obj.type == "MESH" and obj.name != "Icosphere":
                    mesh_object_list.append(obj.name)
            sorted(mesh_object_list)

            for obj_name in mesh_object_list:
                obj = bpy.data.objects[obj_name]
                mesh = obj.data  # Get mesh data
                cur_points_glb = []
                cur_faces_glb = []
                # Ensure the object is in the correct mode to access vertex normals
                bpy.context.view_layer.objects.active = obj
                bpy.ops.object.mode_set(
                    mode="OBJECT"
                )  # Ensure object mode for safe access
                # Get vertex positions and normals
                for vert in mesh.vertices:
                    vert_co_world = obj.matrix_world @ vert.co
                    # vert_co_world = vert.co
                    cur_points_glb.append(
                        vert_co_world[:]
                    )  # Store vertex position as tuple (x, y, z)
                    full_points_glb.append(
                        vert_co_world[:]
                    )  # Store vertex position as tuple (x, y, z)
                for face in mesh.polygons:
                    cur_faces_glb.append(
                        face.vertices[:]
                    )  # Store face vertices as list [v1, v2, v3]
                cur_points_glb = np.array(cur_points_glb).astype(np.float32)
                cur_points_glb = np.dot(
                    cur_points_glb, np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
                )
                cur_faces_glb = np.array(cur_faces_glb).astype(np.int32)
                mesh = trimesh.Trimesh(vertices=cur_points_glb, faces=cur_faces_glb)
                trimesh_list.append(mesh)
            # Merge all trimeshes
            mesh = trimesh.util.concatenate(trimesh_list)
            full_points_glb = np.array(full_points_glb).astype(np.float32)
            full_points_glb = np.dot(
                full_points_glb, np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
            )

            # mesh = trimesh.load_mesh(file_path, force="mesh")
            full_points = np.array(mesh.vertices).astype(np.float32)
            full_normals = np.array(mesh.vertex_normals).astype(np.float32)
            ## Random shuffle the vertices
            random_idx = np.arange(full_points.shape[0])
            np.random.shuffle(random_idx)
            full_points = full_points[random_idx]
            full_normals = full_normals[random_idx]
            full_random_idx_inv = np.argsort(random_idx)
            ## Uniformly sample points from surfaces
            points, face_idx = trimesh.sample.sample_surface_even(mesh, 1024)
            points = points.astype(np.float32)
            normals = mesh.face_normals[face_idx].astype(np.float32)
            # pad the number of point to 1024
            replace = 1024 > points.shape[0]
            indices = np.random.choice(points.shape[0], 1024, replace=replace)
            points = points[indices]
            normals = normals[indices]
            face_idx = face_idx[indices]
            normals = normals / np.linalg.norm(normals, axis=1, keepdims=True)
            # # Check if the vertices is more than 1024
            assert (
                points.shape[0] == 1024
            ), f"points.shape[0] should be 1024, but got {points.shape[0]}"

            # rescale to unit sphere
            center = (np.max(points, axis=0) + np.min(points, axis=0)) / 2
            points -= center
            full_points -= center
            full_points_glb -= center
            scale = np.max(np.abs(points))
            points /= scale
            full_points /= scale
            full_points_glb /= scale

            # some fake gt
            joints = np.ones((64, 3), dtype=np.float32)
            parents = np.arange(64, dtype=np.int32)
            skinning_weights = np.zeros((points.shape[0], 64), dtype=np.float32)
            # Load pointcloud
            batch = {
                "pointcloud": torch.from_numpy(points).unsqueeze(0).to(device),
                "joints": torch.from_numpy(joints).unsqueeze(0).to(device),
                "parents": torch.from_numpy(parents).unsqueeze(0).to(device),
                "skinning_weights": torch.from_numpy(skinning_weights)
                .unsqueeze(0)
                .to(device),
                "normals": torch.from_numpy(normals).unsqueeze(0).to(device),
                "scale": torch.tensor([scale]).to(device),
                "center": torch.from_numpy(center).to(device),
                "item_idx": [item_idx],
                "root_idx": torch.tensor([0]).to(device),
                "full_pointcloud": torch.from_numpy(full_points)
                .unsqueeze(0)
                .to(device),
                "full_normals": torch.from_numpy(full_normals).unsqueeze(0).to(device),
            }

            result = model.generate_sequence(
                batch, create_visual=False, save_skeleton=False, compute_loss=False
            )

            # Project from face skinning to vertex skinning
            npz_dict = result["npz_dict"]
            face_skinning = npz_dict["skinning_weights"]  # [N, J]
            face_pts = npz_dict["pointcloud"]
            # convert to actual weight before interpolation
            joints_skinning_normalized_orig = copy.deepcopy(face_skinning)  # [1, J, N]
            joints_skinning_normalized_orig = torch.tensor(
                joints_skinning_normalized_orig
            )
            # For each rows, keep the top k values
            # k = min(3, joints_skinning_normalized_orig.shape[1])
            k = min(5, joints_skinning_normalized_orig.shape[1])
            _, indices = torch.topk(joints_skinning_normalized_orig, k=k, dim=1)
            joints_skinning_normalized = (
                torch.ones_like(joints_skinning_normalized_orig) * -9999
            )
            joints_skinning_normalized.scatter_(
                1, indices, joints_skinning_normalized_orig.gather(1, indices)
            )
            joints_skinning_normalized = F.softmax(joints_skinning_normalized, dim=1)
            joints_skinning_normalized[joints_skinning_normalized < 0.068] = 0
            # joints_skinning_normalized[joints_skinning_normalized < 0.02] = 0
            skinning_weights = joints_skinning_normalized / (
                joints_skinning_normalized.sum(dim=1, keepdim=True) + 1e-6
            )
            vert_skinning = skinning_weights.cpu().numpy()  # Normalize weights
            # reverse back from random shuffle
            vert_skinning = vert_skinning[full_random_idx_inv]
            vert_pts = full_points[full_random_idx_inv]

            # Smooth skinning
            vert_skinning = smooth_weights_per_vertex(
                mesh, vert_skinning, iterations=10, neighbor_factor=0.35
            )

            # Project from vert_skinning to vert_skinning_glb (glb may have duplicates)
            vert_skinning_glb = project_to_glb(full_points_glb, vert_pts, vert_skinning)
            vert_skinning_glb /= vert_skinning_glb.sum(axis=1, keepdims=True) + 1e-6

            npz_dict["mesh_list"] = mesh_object_list
            npz_dict["skinning_weights"] = vert_skinning_glb
            npz_dict["pointcloud"] = full_points_glb * scale + center
            result["npz_dict"] = npz_dict
            model.save_results(
                batch_dir, result, batch, save_all=True, steps=idx, save_skeleton=True
            )
        torch.cuda.empty_cache()
    exit(0)
