# RigAnything: Template‑Free Autoregressive Rigging for Diverse 3D Assets (SIGGRAPH TOG 2025)

[![Paper](https://img.shields.io/badge/Paper-A42C25?style=for-the-badge&logo=arxiv&logoColor=white)](https://arxiv.org/abs/2502.09615)
[![Project Page](https://img.shields.io/badge/Project%20Page-000000?style=for-the-badge&logo=githubpages&logoColor=white)](https://www.liuisabella.com/RigAnything/)
[![GitHub](https://img.shields.io/badge/GitHub-181717?style=for-the-badge&logo=github&logoColor=white)](https://github.com/Isabella98Liu/RigAnything)
[![Hugging Face Models](https://img.shields.io/badge/Models-fcd022?style=for-the-badge&logo=huggingface&logoColor=000)](https://huggingface.co/Isabellaliu/RigAnything/tree/main)

RigAnything predicts skeletons and skinning for diverse 3D assets without a fixed template. This repository provides inference scripts to rig your meshes (.glb or .obj) end‑to‑end and export a rigged GLB for use in DCC tools (e.g., Blender).

## Environment setup

Recommended: create a fresh Conda env with Python 3.11.

```bash
conda create -n riganything -y python=3.11
conda activate riganything
```

Install PyTorch per your CUDA/CPU setup (see https://pytorch.org/get-started/locally/). Example (adjust CUDA version as needed):

```bash
# GPU example (CUDA 12.x) — pick the right wheel from PyTorch website

# 1) Install PyTorch that matches your system
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126

# 2) Install project dependencies
pip install -r requirements.txt
```

Notes
- The scripts import Blender’s Python API (`bpy`). The `bpy` PyPI package works in headless environments; alternatively, you may use a system Blender installation. If you run into OpenGL/GLX issues on a server, consider an off‑screen setup (e.g., OSMesa/Xvfb) and ensure libGL is available.
- `open3d`/`pymeshlab` may require system GL libraries on Linux (e.g., `libgl1`).

## Checkpoint

Download the pre‑trained checkpoint and place it under `ckpt/`.

```
hf download Isabellaliu/RigAnything --local-dir ckpt/
```

## Quick start

Use the provided script to simplify your mesh (optional) and run inference. The tool accepts either `.glb` or `.obj` as input.

```bash
sh scripts/inference.sh <path_to_mesh.(glb|obj)> <simplify_flag: 0|1> <target_face_count>
```

Example:

```bash
sh scripts/inference.sh data_examples/spyro_the_dragon.glb 1 8192
```

### What the arguments mean
- mesh_path: path to your input mesh (.glb or .obj)
- simplify_flag: whether to simplify the mesh before rigging (0 = no, 1 = yes)
- target_face_count: the target number of faces after simplification (only used when simplify_flag = 1)

### Outputs
Results are written under `outputs/<asset_name>/` with these key files:
- `<name>_simplified.glb`: the simplified input mesh used for inference
- `<name>_simplified.npz`: intermediate results (joints, weights, etc.)
- `<name>_simplified_rig.glb`: the final rigged mesh you can import into Blender
- `inference.log`: logs from all steps

## Advanced: run inference directly

You can call the Python entry points used by the script. Minimal example equivalent to the shell script flow:

```bash
# 1) Optional: simplify
python inference_utils/mesh_simplify.py \
  --data_path data_examples/spyro_the_dragon.glb \
  --mesh_simplify 1 \
  --simplify_count 8192 \
  --output_path outputs/spyro_the_dragon

# 2) Inference (uses config.yaml + checkpoint)
python inference.py \
  --config config.yaml \
  --load ckpt/riganything_ckpt.pt \
  -s inference true \
  -s inference_out_dir outputs/spyro_the_dragon \
  --mesh_path outputs/spyro_the_dragon/spyro_the_dragon_simplified.glb

# 3) Visualize / export rigged GLB
python inference_utils/vis_skel.py \
  --data_path outputs/spyro_the_dragon/spyro_the_dragon_simplified.npz \
  --save_path outputs/spyro_the_dragon \
  --mesh_path outputs/spyro_the_dragon/spyro_the_dragon_simplified.glb
```

## Supported inputs
- `.glb` is supported directly.
- `.obj` is supported and will be converted to `.glb` internally (without textures).

## Tips & troubleshooting
- GPU memory: inference uses the first CUDA device (`cuda:0`). Ensure sufficient VRAM; otherwise consider simplifying the mesh (higher simplification ratio / lower face count).
- Headless servers: if `bpy` complains about display/GL, install the necessary GL libraries and/or use an off‑screen context. Using the `bpy` PyPI wheel typically helps for server environments.

## Citation

If you find this work useful, please cite:

```
@article{liu2025riganything,
  title   = {RigAnything: Template-free autoregressive rigging for diverse 3D assets},
  author  = {Liu, Isabella and Xu, Zhan and Wang, Yifan and Tan, Hao and Xu, Zexiang and Wang, Xiaolong and Su, Hao and Shi, Zifan},
  journal = {ACM Transactions on Graphics (TOG)},
  volume  = {44},
  number  = {4},
  pages   = {1--12},
  year    = {2025},
  publisher = {ACM}
}
```

---

Questions or issues? Please open a GitHub issue or reach out via the project page.
