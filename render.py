import os
os.environ["NUMEXPR_MAX_THREADS"] = "24"
import sys
from argparse import ArgumentParser
from libs.config import parse_args
from libs.render.blender import render
from libs.render.blender.tools import mesh_detect

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

def render_cli() -> None:
    # parse options
    cfg = parse_args(phase="render")  # parse config file
    cfg.FOLDER = cfg.RENDER.FOLDER

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
        jointstype=cfg.RENDER.JOINT_TYPE.lower(),
        res=cfg.RENDER.RES,
        init=init,
        gt=cfg.RENDER.GT,
        accelerator=cfg.ACCELERATOR,
        device=cfg.DEVICE,
    )

if __name__ == "__main__":
    render_cli()
