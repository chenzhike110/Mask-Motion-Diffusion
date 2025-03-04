import bpy
import numpy as np
from .materials import plane_mat  # noqa

def drawLine(coords):
    # create the Curve Datablock
    curveData = bpy.data.curves.new('myCurve', type='CURVE')
    curveData.dimensions = '3D'
    curveData.resolution_u = 2

    # map coords to spline
    polyline = curveData.splines.new('POLY')
    polyline.points.add(len(coords)-1)
    for i, coord in enumerate(coords):
        x,y,z = coord
        polyline.points[i].co = (x, y, z, 1)

    # create Object
    curveOB = bpy.data.objects.new('myCurve', curveData)
    curveData.bevel_depth = 0.01

    material = bpy.data.materials.new('myCurve'+"_material")
    material.diffuse_color = (1.0,0.0,0.0,1.0)
    curveData.materials.append(material)

    # attach to scene and validate context
    bpy.context.collection.objects.link(curveOB)

def setup_renderer(denoising=True, oldrender=True, accelerator="gpu", device=[0]):
    bpy.context.scene.render.engine = "CYCLES"
    bpy.data.scenes[0].render.engine = "CYCLES"
    if accelerator.lower() == "gpu":
        bpy.context.preferences.addons[
            "cycles"
        ].preferences.compute_device_type = "CUDA"
        bpy.context.scene.cycles.device = "GPU"
        i = 0
        bpy.context.preferences.addons["cycles"].preferences.get_devices()
        for d in bpy.context.preferences.addons["cycles"].preferences.devices:
            if i in device:  # gpu id
                d["use"] = 1
                print(d["name"], "".join(str(i) for i in device))
            else:
                d["use"] = 0
            i += 1

    if denoising:
        bpy.context.scene.cycles.use_denoising = True

    # bpy.context.scene.render.tile_x = 256
    # bpy.context.scene.render.tile_y = 256
    bpy.context.scene.cycles.samples = 64
    # bpy.context.scene.cycles.denoiser = 'OPTIX'

    if not oldrender:
        bpy.context.scene.view_settings.view_transform = "Standard"
        bpy.context.scene.render.film_transparent = True
        bpy.context.scene.display_settings.display_device = "sRGB"
        bpy.context.scene.view_settings.gamma = 1.2
        bpy.context.scene.view_settings.exposure = -0.75


# Setup scene
def setup_scene(
    res="high", denoising=True, oldrender=True, accelerator="gpu", device=[0], scene_mesh=None
):
    scene = bpy.data.scenes["Scene"]
    assert res in ["ultra", "high", "med", "low"]
    if res == "high":
        scene.render.resolution_x = 1280
        scene.render.resolution_y = 1024
    elif res == "med":
        scene.render.resolution_x = 1280 // 2
        scene.render.resolution_y = 1024 // 2
    elif res == "low":
        scene.render.resolution_x = 1280 // 4
        scene.render.resolution_y = 1024 // 4
    elif res == "ultra":
        scene.render.resolution_x = 1280 * 2
        scene.render.resolution_y = 1024 * 2

    scene.render.film_transparent= True
    world = bpy.data.worlds["World"]
    world.use_nodes = True
    bg = world.node_tree.nodes["Background"]
    bg.inputs[0].default_value[:3] = (1.0, 1.0, 1.0)
    bg.inputs[1].default_value = 1.0

    # Remove default cube
    if "Cube" in bpy.data.objects:
        bpy.data.objects["Cube"].select_set(True)
        bpy.ops.object.delete()

    bpy.ops.object.light_add(
        type="SUN", align="WORLD", location=(0, 0, 0), scale=(1, 1, 1)
    )
    bpy.data.objects["Sun"].data.energy = 1.5

    # rotate camera
    bpy.ops.object.empty_add(
        type="PLAIN_AXES", align="WORLD", location=(0, 0, 0), scale=(1, 1, 1)
    )
    bpy.ops.transform.resize(
        value=(10, 10, 10),
        orient_type="GLOBAL",
        orient_matrix=((1, 0, 0), (0, 1, 0), (0, 0, 1)),
        orient_matrix_type="GLOBAL",
        mirror=True,
        use_proportional_edit=False,
        proportional_edit_falloff="SMOOTH",
        proportional_size=1,
        use_proportional_connected=False,
        use_proportional_projected=False,
    )  
    bpy.ops.object.select_all(action="DESELECT")

    if scene_mesh:
        for key in scene_mesh.keys():
            if key == 'Line':
                line = np.load(scene_mesh[key]['file'])
                drawLine(line)
                bpy.ops.object.select_all(action="DESELECT")
            else:
                bpy.ops.wm.obj_import(filepath=scene_mesh[key]['file'])
                bpy.context.object.scale = scene_mesh[key]['scale']
                bpy.context.object.location = scene_mesh[key]['position']
                bpy.context.object.rotation_euler = scene_mesh[key]['euler']
                bpy.ops.object.select_all(action="DESELECT")

    setup_renderer(
        denoising=denoising, oldrender=oldrender, accelerator=accelerator, device=device
    )
    return scene
