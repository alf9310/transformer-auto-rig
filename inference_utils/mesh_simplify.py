import argparse

import bpy
import numpy as np
import open3d as o3d
import pymeshlab
import trimesh


def create_mesh_object(mesh, mesh_collection=None):
    """Create mesh object and link to scene"""
    obj = bpy.data.objects.new("PointCloudObject", mesh)
    if mesh_collection:
        mesh_collection.objects.link(obj)
    else:
        bpy.context.scene.collection.objects.link(obj)

    # Create material
    mat = bpy.data.materials.new(name="WhiteMaterial")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes

    # Clear default nodes
    nodes.clear()

    # Create a principled BSDF node
    principled = nodes.new("ShaderNodeBsdfPrincipled")
    principled.inputs["Base Color"].default_value = (
        0.566,
        0.862,
        1.0,
        1,
    )  # White color
    principled.inputs["Metallic"].default_value = 0.02

    # Create material output node
    material_output = nodes.new("ShaderNodeOutputMaterial")

    # Link nodes
    mat.node_tree.links.new(
        principled.outputs["BSDF"], material_output.inputs["Surface"]
    )

    # Assign material to object
    obj.data.materials.append(mat)

    # Enable wireframe display
    # obj.show_wire = True  # Show wireframe

    return obj


def mesh_simplify_obj(file_path, simplify=False, simplify_count=4096, output_path="outputs"):
    # CASE 1: FOR (.OBJ), assume no texture
    output_path = file_path.replace(".obj", "_simplified.obj")

    # Method 1 PyMeshLab
    ms = pymeshlab.MeshSet()
    ms.load_new_mesh(file_path)
    if simplify:
        # target_face = int(ms.current_mesh().face_number() * simplify_ratio)
        target_face = min(simplify_count, ms.current_mesh().face_number())
        ms.meshing_decimation_quadric_edge_collapse(
            targetfacenum=target_face,  # Target number of faces
            qualitythr=0.8,  # Quality threshold
            preservetopology=True,  # Preserve topology
            preserveboundary=True,  # Preserve boundary vertices
            boundaryweight=1.0,  # Weight of boundary vertices
            optimalplacement=True,  # Better vertex placement
            preservenormal=True,  # Preserve normals
            planarquadric=False,  # Use planar quadric
            planarweight=0.8,  # Weight of planar quadric
            autoclean=True,  # Clean mesh after simplification
        )
        print("Mesh simplified and saved to", output_path)
    verts = ms.current_mesh().vertex_matrix()
    # Rotate the mesh to make it upright
    verts = np.dot(verts, np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]]))

    mesh = trimesh.Trimesh(vertices=verts, faces=ms.current_mesh().face_matrix())
    mesh.export(output_path)

    # Convert to glb file format
    # Set up rendering engine
    bpy.context.scene.render.engine = "CYCLES"
    # Use GPU
    bpy.context.scene.cycles.device = "GPU"
    # Change color management to standard
    bpy.context.scene.view_settings.view_transform = "Standard"
    # Clear existing scene
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()
    blender_mesh = bpy.data.meshes.new("PointCloudMesh")

    # Normalize the mesh to fit in the unit sphere
    vert = mesh.vertices
    vert = vert - np.mean(vert, axis=0)
    vert = vert / np.max(np.linalg.norm(vert, axis=1))
    mesh.vertices = vert

    blender_mesh.from_pydata(
        mesh.vertices.tolist(), [], mesh.faces.tolist()
    )  # For PyMeshLab
    # blender_mesh.vertex_colors.new(name="Col")
    # for i, f in enumerate(mesh.faces):
    #     for j in range(3):
    #         blender_mesh.vertex_colors["Col"].data[i].color[j] = mesh.visual.vertex_colors[0].data[i][j]
    # blender_mesh.from_pydata(mesh.vertices, [], mesh.triangles)  # For Open3D
    blender_mesh.update()
    mesh_obj = create_mesh_object(blender_mesh)
    # Export to glb
    mesh_obj.select_set(True)
    mesh_glb_path = output_path + "/" + file_path.split("/")[-1].replace(".obj", "_simplified.glb")
    bpy.ops.export_scene.gltf(
        filepath=mesh_glb_path,
        export_format="GLB",  # Must be 'GLB' or 'GLTF_SEPARATE' or 'GLTF_EMBEDDED'
        use_selection=True,  # Must be True or False
        # export_draco_mesh_compression_enable=True,  # Must be True or False
        # Other parameters...
        export_materials="EXPORT",  # Use string enum instead of bool
        export_apply=False,
    )
    print("Mesh exported to", mesh_glb_path)


def mesh_simplify_glb(file_path, simplify=False, simplify_count=4096, output_path="outputs"):
    # CASE 2: FOR (.GLB)
    if simplify:
        # Empty blender scene
        bpy.ops.object.select_all(action="SELECT")
        bpy.ops.object.delete()
        # Import glb file
        bpy.ops.import_scene.gltf(filepath=file_path)

        total_faces = 0
        for obj in bpy.data.objects:
            if obj.type == "MESH":
                total_faces += len(obj.data.polygons)
        print("Total faces:", total_faces)
        simplify_ratio = simplify_count / total_faces
        simplify_ratio = np.clip(simplify_ratio, 0.01, 1.0)
        # Loop through all mesh object in the scene and perform simplification
        for obj in bpy.data.objects:
            if obj.type == "MESH":
                # obj.select_set(True)
                bpy.context.view_layer.objects.active = obj
                decimate_mod = obj.modifiers.new(name="Decimate", type="DECIMATE")
                decimate_mod.ratio = simplify_ratio
                decimate_mod.decimate_type = "COLLAPSE"
                bpy.ops.object.modifier_apply(modifier=decimate_mod.name)
        # Export to glb
        # mesh_glb_path = file_path.replace(".glb", "_simplified.glb")
        mesh_glb_path = output_path + "/" + file_path.split("/")[-1].replace(".glb", "_simplified.glb")
        bpy.ops.export_scene.gltf(
            filepath=mesh_glb_path,
            export_format="GLB",  # Must be 'GLB' or 'GLTF_SEPARATE' or 'GLTF_EMBEDDED'
            use_selection=False,  # Must be True or False
            # export_draco_mesh_compression_enable=True,  # Must be True or False
            # Other parameters...
            export_materials="EXPORT",  # Use string enum instead of bool
            export_apply=False,
        )
    else:
        mesh_glb_path = output_path + "/" + file_path.split("/")[-1].replace(".glb", "_simplified.glb")
        bpy.ops.wm.copy_file(filepath=file_path, target=mesh_glb_path)
    print("Mesh exported to", mesh_glb_path)


def main():
    args = argparse.ArgumentParser()
    args.add_argument("--data_path", type=str, default="", required=True)
    args.add_argument("--mesh_simplify", type=int, default=0)
    args.add_argument("--simplify_count", type=int, default=4096)
    args.add_argument("--output_path", type=str, default="outputs")
    args = args.parse_args()
    data_path = args.data_path

    if args.data_path.endswith(".obj"):
        mesh_simplify_obj(data_path, args.mesh_simplify, args.simplify_count, args.output_path)
    elif args.data_path.endswith(".glb"):
        mesh_simplify_glb(data_path, args.mesh_simplify, args.simplify_count, args.output_path)


if __name__ == "__main__":
    main()
