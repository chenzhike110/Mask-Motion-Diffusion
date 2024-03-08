import os
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
import torch
import numpy as np
from transformers import AutoModel, AutoTokenizer
from libs.config import parse_args
from libs.get_model import get_model_with_config, get_dataset
from libs.data.utils import generate_sin
from libs.losses.geodesic_loss import geodesic_loss_R
from pytorch3d.transforms import euler_angles_to_matrix, matrix_to_rotation_6d, axis_angle_to_matrix, rotation_6d_to_matrix, matrix_to_euler_angles, matrix_to_axis_angle

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

from libs.body_model import BodyModel

device = torch.device('cuda:1')
body_model = BodyModel("deps/body_models/smplh/neutral/model.npz").to(device)
trajectories = []

def pelvis_follows(length, x_forward=torch.pi*3., y_width=1.0):

    pos, orient = generate_sin(length, x_forward, y_width, device)
    vel = pos[1:] - pos[:-1]
    vel = torch.cat([torch.zeros(1, 3).to(device), vel]).unsqueeze(0)

    root_orient = torch.tensor([[torch.pi/2., -torch.pi/2., 0.]]).repeat(length, 1).to(device)
    root_orient[:, 1:2] += orient
    root_orient = euler_angles_to_matrix(root_orient, 'XYZ')

    trajectories.append(pos[..., :2])

    def constraint(sample):
        root_pos = torch.cumsum(sample[..., :2], dim=-2).squeeze(0)
        pos_loss = torch.nn.L1Loss()(root_pos, pos[..., :2])
        # orient_loss = geodesic_loss_R()(rotation_6d_to_matrix(sample[..., 3:9]).squeeze(0), root_orient)
        orient_loss = 0.
        trajectories.append(root_pos.detach().cpu())
        return orient_loss + pos_loss
    
    return constraint

def generate_sit_interaction(length=80):
    from scipy import signal
    from libs.interaction import Interactions
    guassian_range = 5
    weights = signal.windows.gaussian(guassian_range*2-1, std=3.)
    
    targets = Interactions('datasets/BEHAVE/distribution.npy', device=device)
    target = targets.humanpose[0, None]
    
    sample_init = torch.tensor([1., 0., 0., 0., 1., 0.]).reshape(1, 1, 6).repeat(1, length, 22)
    sample_init[:, :, :6] = matrix_to_rotation_6d(
        euler_angles_to_matrix(torch.tensor([[torch.pi/2., -torch.pi/4., 0.]]), 'XYZ')
    )
    
    sit_pos = (torch.rand(3) - 0.5) * 6
    sit_pitch = (torch.rand(1) - 0.5) * 2 * torch.pi
    sit_pos[-1] = 0.45
    print(sit_pitch, sit_pos)
    
    sit_trans = torch.eye(4)
    sit_trans[:-1, :-1] = euler_angles_to_matrix(torch.tensor([[torch.pi/2., sit_pitch, 0.]]), 'XYZ')
    sit_trans[:-1, -1] = sit_pos
    
    root_trans, root_rot, pose_body = targets.object_to_smpl(target, sit_trans[None])
    
    target_rotation = axis_angle_to_matrix(torch.cat([root_rot, target[:, 6:69]], dim=-1).reshape(-1, 3)).float()
    
    sample_init[0, -guassian_range] = matrix_to_rotation_6d(target_rotation).flatten()
    sample_init = torch.cat([torch.zeros((1, length, 3)), sample_init], dim=-1).to(device)
    
    noise_mask = torch.zeros((length))
    noise_mask[-guassian_range:] = torch.from_numpy(weights[:guassian_range])
    noise_mask = noise_mask.unsqueeze(-1).repeat(1, 22*6)
    noise_mask = torch.cat([torch.zeros((length, 3)), noise_mask], dim=-1).to(device)
    
    traj_0 = root_trans[0, :2].repeat(length, 1).cpu().numpy()
    traj_0[0] = 0
    trajectories.append(traj_0)
    
    def constraint(sample):
        root_pos = torch.cumsum(sample[..., :2], dim=-2).squeeze(0)
        root_pos = torch.cat([root_pos, sample[..., 2:3].squeeze(0)], dim=-1)
        pos_loss = torch.nn.L1Loss()(root_pos[-guassian_range:], root_trans.repeat(guassian_range, 1))
        orient_loss = geodesic_loss_R()(rotation_6d_to_matrix(sample[0, -1, 3:].reshape(-1, 6)), target_rotation)
        trajectories.append(root_pos[..., :2].detach().cpu())
        return pos_loss + orient_loss * 0.1
    
    return constraint, noise_mask, sample_init

def main():
    cfg = parse_args()
    datamodule = get_dataset(cfg)
    model = get_model_with_config(cfg, datamodule)
    model.load_state_dict(torch.load(cfg.MODEL.CHECKPOINT)['state_dict'])
    model = model.to(device)
    model.eval()

    text_encoder = AutoModel.from_pretrained('./deps/clip-vit-base-patch16').to(device)
    tokenizer = AutoTokenizer.from_pretrained('./deps/clip-vit-base-patch16')

    with torch.no_grad():
        text_inputs = tokenizer(
            ["a person walks to a seat, then sits on it."], 
            padding="max_length",
            truncation=True,
            max_length=tokenizer.model_max_length,
            return_tensors="pt"
        )
        text_input_ids = text_inputs.input_ids
        caption = text_encoder.get_text_features(
            text_input_ids.to(text_encoder.device)
        ).cpu().numpy()

    length = 80

    def hand_follows(length, x_forward=2., heights=0.9):
        from libs.body_model.joint_names import SMPLH_BONE_ORDER_NAMES
        x = torch.linspace(0, -x_forward, length).unsqueeze(-1).to(device)
        hand_pos = torch.cat([x, torch.ones_like(x)*0.08, torch.ones_like(x)*heights], dim=-1)
        root_pos_ref = torch.cat([x, -torch.ones_like(x) * 0.25, torch.ones_like(x)*heights], dim=-1)
        trajectories.append(hand_pos[..., [1,2]])
        
        noise_mask = torch.zeros((length, 22, 1))
        mask_joint_name = ["R_Shoulder"]
        mask_joint_index = [SMPLH_BONE_ORDER_NAMES.index(i) for i in mask_joint_name]
        noise_mask[:, mask_joint_index, :] = 1.0
        noise_mask = noise_mask.repeat(1, 1, 6).flatten(start_dim=1)
        noise_mask = torch.cat([torch.zeros((length, 3)), noise_mask], dim=-1).bool()

        def constraint(sample):
            root_pos = torch.cumsum(sample[..., :2], dim=-2).squeeze(0)
            root_pos_loss = torch.nn.L1Loss()(root_pos, root_pos_ref[..., :2])

            smpls = datamodule.recover_motion(sample, local_only=False, normalized=cfg.MODEL.normalize)
            hands = smpls.Jtr[:, SMPLH_BONE_ORDER_NAMES.index('R_Middle2')]
            pos_loss = torch.nn.L1Loss()(hands[:, 1:], hand_pos[:, 1:])
            trajectories.append(hands[:, [1,2]].detach().cpu())
            return pos_loss * 10. + root_pos_loss
        
        return constraint, noise_mask, None

    constraint, noise_mask, latent_init = generate_sit_interaction(length)

    with torch.no_grad():
        smpls = model.sample_with_constraints(
            batch={"text":[caption], "length":[length]},
            constraint=constraint,
            noise_mask=noise_mask,
            latent_init=latent_init
        ).cpu()
        smpls = datamodule.recover_motion(smpls, local_only=False, normalized=cfg.MODEL.normalize)
        np.save('./demos/samples/edit_sit_chair.npy', smpls.v.numpy())

    # save prograss
    fig, ax = plt.subplots()
    def animate(i):
        ax.clear()
        line, = ax.plot(trajectories[(i+1)*10][:, 0], trajectories[(i+1)*10][:, 1], color = 'blue', lw=1)
        line2, = ax.plot(trajectories[0][:, 0], trajectories[0][:, 1], color = 'red', lw=1)
        return line, line2
    ani = FuncAnimation(fig, animate, interval=40, blit=True, repeat=True, frames=(len(trajectories)-1)//10)    
    ani.save("TLI.gif", dpi=300, writer=PillowWriter(fps=25))

if __name__ == "__main__":
    main()