import os
os.environ["NUMEXPR_MAX_THREADS"] = "24"
import sys
# sys.path.append("/home/dhz/miniconda3/envs/czk_render/lib/python3.7/site-packages")

import random
import shutil
import numpy as np
from pathlib import Path

import natsort

try:
    import bpy
    sys.path.append(os.path.dirname(bpy.data.filepath))
except ImportError:
    raise ImportError(
        "Blender is not properly installed or not launch properly. See README.md to have instruction on how to install and use blender."
    )
from argparse import ArgumentParser

# Monkey patch argparse such that
# blender / python / hydra parsing works
def parse_arg(self, args=None, namespace=None):
    if args is not None:
        return self.parse_args_bak(args=args, namespace=namespace)
    try:
        idx = sys.argv.index("--")
        args = sys.argv[idx+1:]  # the list after '--'
    except ValueError as e:  # '--' not in the list:
        args = []
    return self.parse_args_bak(args=args, namespace=namespace)

setattr(ArgumentParser, 'parse_args_bak', ArgumentParser.parse_args)
setattr(ArgumentParser, 'parse_args', parse_arg)

from libs.config import parse_args
from libs.render.blender import render
from libs.render.video import Video

def render_cli() -> None:
    # parse options
    cfg = parse_args(phase="render")  # parse config file
    
    if cfg.RENDER.INPUT_MODE.lower() == "npy":
        output_dir = Path(os.path.dirname(cfg.RENDER.NPY))
        paths = [cfg.RENDER.NPY]
        # print("xxx")
        # print("begin to render for{paths[0]}")
    elif cfg.RENDER.INPUT_MODE.lower() == "dir":
        output_dir = Path(cfg.RENDER.DIR)
        paths = []
        # file_list = os.listdir(cfg.RENDER.DIR)
        # random begin for parallel
        file_list = natsort.natsorted(os.listdir(cfg.RENDER.DIR))
        begin_id = random.randrange(0, len(file_list))
        file_list = file_list[begin_id:]+file_list[:begin_id]

        # render mesh npy first
        for item in file_list:
            if item.endswith("_mesh.npy"):
                paths.append(os.path.join(cfg.RENDER.DIR, item))

        # then render other npy
        for item in file_list:
            if item.endswith(".npy") and not item.endswith("_mesh.npy"):
                paths.append(os.path.join(cfg.RENDER.DIR, item))

        print(f"begin to render for {paths[0]}")

    init = True
    for path in paths:
        # check existed mp4 or under rendering
        if cfg.RENDER.MODE == "video":
            if not cfg.override:
                if os.path.exists(path.replace(".npy", ".mp4")) or os.path.exists(path.replace(".npy", "_frames")):
                    print(f"npy is rendered or under rendering {path}")
                    continue
            frames_folder = os.path.join(
                output_dir, path.replace(".npy", "_frames").split('/')[-1])
            os.makedirs(frames_folder, exist_ok=True)
        else:
            # check existed png
            if not cfg.override:
                if os.path.exists(path.replace(".npy", ".png")):
                    print(f"npy is rendered or under rendering {path}")
                    continue
            frames_folder = os.path.join(
                output_dir, path.replace(".npy", ".png").split('/')[-1])
            
        data = np.load(path)

        scene = None
        if hasattr(cfg, "MESH"):
            scene = cfg.MESH
            for key in scene.keys():
                if not hasattr(scene[key], 'file'):
                    scene[key]['file'] = cfg[key]

        out = render(
            data,
            frames_folder,
            canonicalize=cfg.RENDER.CANONICALIZE,
            exact_frame=cfg.RENDER.EXACT_FRAME,
            num=cfg.RENDER.NUM,
            mode=cfg.RENDER.MODE,
            faces_path=cfg.RENDER.FACES_PATH,
            downsample=cfg.RENDER.DOWNSAMPLE,
            always_on_floor=cfg.RENDER.ALWAYS_ON_FLOOR,
            oldrender=cfg.RENDER.OLDRENDER,
            jointstype='HumanML3D'.lower(),
            res=cfg.RENDER.RES,
            init=init,
            gt=cfg.RENDER.GT,
            accelerator=cfg.ACCELERATOR,
            device=cfg.DEVICE,
            scene=scene
        )

        init = False

        if cfg.RENDER.MODE == "video":
            if cfg.RENDER.DOWNSAMPLE:
                video = Video(frames_folder, fps=cfg.RENDER.FPS)
            else:
                video = Video(frames_folder, fps=cfg.RENDER.FPS)

            vid_path = frames_folder.replace("_frames", ".mp4")
            video.save(out_path=vid_path)
            shutil.rmtree(frames_folder)
            print(f"remove tmp fig folder and save video in {vid_path}")

        else:
            print(f"Frame generated at: {out}")

if __name__ == "__main__":
    render_cli()
