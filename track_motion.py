import argparse

import sys
import os.path as osp
import os

os.environ["OMP_NUM_THREADS"] = "1"
sys.path.append(os.getcwd())

import torch
import numpy as np

from libs.physics.utils.copycat_config import Config

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", default=None)
    parser.add_argument("--test", action="store_true", default=False)
    parser.add_argument("--num_threads", type=int, default=30)
    parser.add_argument("--gpu_index", type=int, default=0)
    parser.add_argument("--epoch", type=int, default=0)
    parser.add_argument("--show_noise", action="store_true", default=False)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--data", type=str, default="sample_data/amass_copycat_take5_test_small.pkl")
    parser.add_argument("--mode", type=str, default="vis")
    parser.add_argument("--render_video", action="store_true", default=False)
    parser.add_argument("--render_rfc", action="store_true", default=False)
    parser.add_argument("--render", action="store_true", default=False)
    parser.add_argument("--hide_expert", action="store_true", default=False)
    parser.add_argument("--no_fail_safe", action="store_true", default=False)
    parser.add_argument("--focus", action="store_true", default=False)
    parser.add_argument("--output", type=str, default="test")
    parser.add_argument("--shift_expert", action="store_true", default=False)
    parser.add_argument("--smplx", action="store_true", default=False)
    parser.add_argument("--hide_im", action="store_true", default=False)
    parser.add_argument("--record", action="store_true", default=False)
    args = parser.parse_args()

    cfg = Config(cfg_id=args.cfg, create_dirs=False)
    cfg.update(args)
    
    cfg.no_log = True
    if args.no_fail_safe:
        cfg.fail_safe = False

    cfg.output = osp.join("results/renderings/uhc/", f"{cfg.id}")
    os.makedirs(cfg.output, exist_ok=True)

    cfg.data_specs["file_path"] = args.data

    if "test_file_path" in cfg.data_specs:
        del cfg.data_specs["test_file_path"]

    if cfg.mode == "vis":
        cfg.num_threads = 1

    dtype = torch.float64
    torch.set_default_dtype(dtype)
    # device = (
    #     torch.device("cuda", index=args.gpu_index)
    #     if torch.cuda.is_available()
    #     else torch.device("cpu")
    # )
    device = torch.device("cpu")
    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu_index)
    print(f"Using: {device}")
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    if cfg.smplx and cfg.robot_cfg["model"] == "smplh":
        cfg.robot_cfg["model"] = "smplx"

    from libs.physics.agents.agent_track import AgentTrack

    agent = AgentTrack(cfg, dtype, device, training=True, checkpoint_epoch=args.epoch)

    if args.mode == "stats":
        agent.eval_policy(epoch=args.epoch, dump=True)
    elif args.mode == "disp_stats":
        from libs.physics.render.copycat_visualizer import CopycatVisualizer

        vis = CopycatVisualizer(agent.env.smpl_robot.export_vis_string().decode("utf-8"), agent)
        vis.display_coverage()
    else:
        from libs.physics.render.copycat_visualizer import CopycatVisualizer

        vis = CopycatVisualizer(agent.env.smpl_robot.export_vis_string().decode("utf-8"), agent, cfg.record)
        vis.show_animation()
        if cfg.record:
            vis.end_record()
