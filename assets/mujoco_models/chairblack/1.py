import trimesh

mesh = trimesh.load('assets/mujoco_models/chairblack/decomp.obj')
mesh.export('assets/mujoco_models/chairblack/decomp.stl')