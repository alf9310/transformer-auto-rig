#!/bin/bash

# Usage: ./inference.sh <mesh_path> <mesh_simplify (true/false)> <simplify_count>

# Check if an argument is provided
if [ $# -lt 3 ]; then
    echo "Error: Please provide mesh_simplify and simplify_count as arguments"
    echo "Usage: $0 <mesh_path> <mesh_simplify (0/1)> <simplify_count>"
    exit 1
fi

# Store the mesh path argument
DATA_PATH="$1"
MESH_SIMPLIFY="$2"
MESH_SIMPLIFY_COUNT="$3"
DATA_NAME="${DATA_PATH##*/}"

# Step 0: Create output directory
OUTPUT_DIR="outputs/${DATA_NAME%.glb}/"
mkdir -p "$OUTPUT_DIR"
INFERENCE_LOG="$OUTPUT_DIR/inference.log"
touch "$INFERENCE_LOG"

# Step 1: Run mesh simplification if specified
echo "---------------------------Step 1: Mesh Simplification---------------------------"
echo "Executing: python inference_utils/mesh_simplify.py --data_path $DATA_PATH --mesh_simplify $MESH_SIMPLIFY --simplify_count $MESH_SIMPLIFY_COUNT --output_path $OUTPUT_DIR   "
echo " "
python inference_utils/mesh_simplify.py \
    --data_path "$DATA_PATH" \
    --mesh_simplify "$MESH_SIMPLIFY" \
    --simplify_count "$MESH_SIMPLIFY_COUNT" \
    --output_path "$OUTPUT_DIR" >> "$INFERENCE_LOG" 2>&1

MESH_SIMPLIFIED_PATH="$OUTPUT_DIR/${DATA_NAME%.glb}_simplified.glb"


# Step 2: Run RigAnything inference
echo "--------------------------Step 2: RigAnything Inference---------------------------"
echo "Executing: python inference.py --config config.yaml --load ckpt/riganything_ckpt.pt -s inference true -s inference_out_dir outputs --mesh_path $MESH_SIMPLIFIED_PATH"
echo " "
python inference.py \
    --config config.yaml \
    --load ckpt/riganything_ckpt.pt \
    -s inference true \
    -s inference_out_dir $OUTPUT_DIR \
    --mesh_path "$MESH_SIMPLIFIED_PATH" >> "$INFERENCE_LOG" 2>&1

INFERENCE_OUTPUT_NPZ_PATH="$OUTPUT_DIR/${DATA_NAME%.glb}_simplified.npz"

# # Step 3: Run visualization
echo "---------------------------Step 3: Visualization----------------------------------"
echo "Executing: python inference_utils/vis_skel.py --data_path $INFERENCE_OUTPUT_NPZ_PATH --save_path $OUTPUT_DIR --mesh_path $MESH_SIMPLIFIED_PATH"
echo "---------------------------------------------------------------------------------"
echo " "
python inference_utils/vis_skel.py \
    --data_path "$INFERENCE_OUTPUT_NPZ_PATH" \
    --save_path "$OUTPUT_DIR" \
    --mesh_path "$MESH_SIMPLIFIED_PATH" >> "$INFERENCE_LOG" 2>&1

RESULTS_PATH="${INFERENCE_OUTPUT_NPZ_PATH%.npz}_rig.glb"

echo "---------------------------------------------------------------------------------"
echo "Finished! Results saved to $RESULTS_PATH, import with Blender to view the rigged mesh."