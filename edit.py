import torch
import numpy as np
from transformers import AutoModel, AutoTokenizer
from libs.config import parse_args
from libs.get_model import get_model_with_config, get_dataset
from libs.data.utils import generate_sin
from libs.losses.geodesic_loss import geodesic_loss_R
from pytorch3d.transforms import euler_angles_to_matrix, rotation_6d_to_matrix

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

from libs.body_model import BodyModel

device = torch.device('cpu')
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
            ["a person walks forward."], 
            padding="max_length",
            truncation=True,
            max_length=tokenizer.model_max_length,
            return_tensors="pt"
        )
        text_input_ids = text_inputs.input_ids
        caption = text_encoder.get_text_features(
            text_input_ids.to(text_encoder.device)
        ).cpu().numpy()

    length = 120

    def hand_follows(length, x_forward=2., heights=0.88):
        from libs.body_model.joint_names import SMPLH_BONE_ORDER_NAMES
        x = torch.linspace(0, -x_forward, length).unsqueeze(-1).to(device)
        hand_pos = torch.cat([x, torch.zeros_like(x), torch.ones_like(x)*heights], dim=-1)
        root_pos_ref = torch.cat([x, -torch.ones_like(x) * 0.4, torch.ones_like(x)*heights], dim=-1)
        trajectories.append(hand_pos[..., [1,2]])

        def constraint(sample):
            root_pos = torch.cumsum(sample[..., :2], dim=-2).squeeze(0)
            root_pos_loss = torch.nn.L1Loss()(root_pos, root_pos_ref[..., :2])

            smpls = datamodule.recover_motion(sample, local_only=False, normalized=cfg.MODEL.normalize)
            hands = smpls.Jtr[:, SMPLH_BONE_ORDER_NAMES.index('R_Middle2')]
            pos_loss = torch.nn.L1Loss()(hands[:, 1:], hand_pos[:, 1:])
            trajectories.append(hands[:, [1,2]].detach().cpu())
            return pos_loss * 5. + root_pos_loss
        
        return constraint

    constraint = hand_follows(length)

    with torch.no_grad():
        smpls = model.sample_with_constraints(
            batch={"text":[caption], "length":[length]},
            constraint=constraint,
        ).cpu()
        smpls = datamodule.recover_motion(smpls, local_only=False, normalized=cfg.MODEL.normalize)
        np.save('./demos/samples/edit_hand.npy', smpls.v[..., [0,2,1]].numpy())

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