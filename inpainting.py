import torch
import pickle
import numpy as np
from scipy import signal
from transformers import AutoModel, AutoTokenizer
from libs.config import parse_args
from libs.get_model import get_model_with_config, get_dataset
from pytorch3d.transforms import matrix_to_rotation_6d, euler_angles_to_matrix, axis_angle_to_matrix

device = torch.device('cpu')

def generate_sitcrying(length=50, guassian_range=21):
    """
    sit crying inpainting
    """
    weights = signal.windows.gaussian(guassian_range, std=10.)
    with open("demos/pose_estimation/sit_crying.pkl", 'rb') as f:
        data = pickle.load(f)
    poses = torch.from_numpy(data['thetas'][None, :22*3])
    poses = axis_angle_to_matrix(poses.reshape(-1, 22, 3))
    poses[:, 0, ...] = torch.bmm(euler_angles_to_matrix(torch.tensor([[-torch.pi/2., torch.pi/4., 0.]]), 'XYZ').repeat(1, 1, 1), poses[:, 0, ...])
    poses = matrix_to_rotation_6d(poses).view(-1, 22*6)

    text = ["A person sits crying."]

    inpainting_mask = torch.zeros(length)
    sample = torch.tensor([1., 0., 0., 0., 1., 0.]).reshape(1, 1, 6).repeat(1, length, 22)
    sample[:, :, :6] = matrix_to_rotation_6d(
        euler_angles_to_matrix(torch.tensor([[torch.pi/2., -torch.pi/4., 0.]]), 'XYZ')
    )

    # inpainting_mask[:guassian_range] = torch.from_numpy(weights)
    # sample[:, :guassian_range] = poses

    inpainting_mask[-guassian_range:] = torch.from_numpy(weights)
    sample[:, -guassian_range:] = poses

    inpainting_mask = inpainting_mask.reshape(1, length, 1).repeat(1, 1, 22*6)

    sample = torch.cat([torch.zeros((1, length, 3)), sample], dim=-1)
    inpainting_mask = torch.cat([torch.zeros((1, length, 3)), inpainting_mask], dim=-1)

    save_name = 'demos/samples/demo_sit.npy'
    
    return text, sample, inpainting_mask, save_name

def generate_taiji(cutin=3, length=120, guassian_range=31):
    """
    taiji inpainting with gaussian weights
    """
    weights = signal.windows.gaussian(guassian_range, std=1.)

    poses = []
    for i in range(cutin):
        with open(f"./demos/pose_estimation/smpl_{i}.pkl", 'rb') as f:
            data = pickle.load(f)
        poses.append(data['thetas'][None, :22*3])
    poses = torch.from_numpy(np.concatenate(poses))
    poses = axis_angle_to_matrix(poses.reshape(-1, 22, 3))
    poses[:, 0, ...] = torch.bmm(euler_angles_to_matrix(torch.tensor([[-torch.pi/2., torch.pi/4., 0.]]), 'XYZ').repeat(cutin, 1, 1), poses[:, 0, ...])
    poses = matrix_to_rotation_6d(poses).view(-1, 22*6)
    
    text = ["A person plays Tai Chi."]
    inpainting_mask = torch.zeros(length)

    cutin_index = torch.linspace(0, length-guassian_range-1, cutin).long()
    # inpainting_mask[0:5] = 1.
    # inpainting_mask[-5:] = 1.
    # inpainting_mask[20:30] = 1.
    
    sample = torch.tensor([1., 0., 0., 0., 1., 0.]).reshape(1, 1, 6).repeat(1, length, 22)
    sample[:, :, :6] = matrix_to_rotation_6d(
        euler_angles_to_matrix(torch.tensor([[torch.pi/2., -torch.pi/4., 0.]]), 'XYZ')
    )

    for idx, index in enumerate(cutin_index):
        inpainting_mask[index:index+guassian_range] = torch.from_numpy(weights)
        sample[:, index:index+guassian_range] = poses[idx]

    inpainting_mask = inpainting_mask.reshape(1, length, 1).repeat(1, 1, 22*6)
    inpainting_mask[:, :, :6] += 0.2
    inpainting_mask = torch.clamp(inpainting_mask, 0., 1.)

    sample = torch.cat([torch.zeros((1, length, 3)), sample], dim=-1)
    inpainting_mask = torch.cat([torch.zeros((1, length, 3)), inpainting_mask], dim=-1)

    save_name = 'demos/samples/demo_ddpm.npy'
    
    return text, sample, inpainting_mask, save_name

def generate_pelvis(length=120, x_forward=torch.pi*3., y_width=1.0):
    x = torch.linspace(0, -x_forward, length).unsqueeze(-1)
    y = torch.sin(x) * y_width
    pos = torch.cat([x, y, torch.zeros_like(x)], dim=-1)
    vel = pos[1:] - pos[:-1]
    vel = torch.cat([torch.zeros(1, 3), vel]).unsqueeze(0)

    text = ["a person runs."]
    sample = torch.tensor([1., 0., 0., 0., 1., 0.]).reshape(1, 1, 6).repeat(1, length, 22)
    sample[:, :, :6] = matrix_to_rotation_6d(
        euler_angles_to_matrix(torch.tensor([[torch.pi/2., -torch.pi/2., 0.]]), 'XYZ')
    )
    sample = torch.cat([vel, sample], dim=-1)

    # cutin_index = torch.linspace(0, length-guassian_range-1, cutin).long()

    inpainting_mask = torch.zeros_like(sample)
    inpainting_mask[:, torch.linspace(0, length-1, 7).long(), :2] = 0.8
    inpainting_mask[:, 0, 3:9] = 1.0
    inpainting_mask[:, -1, 3:9] = 1.0
    save_name = 'demos/samples/walk_follow.npy'
    return text, sample, inpainting_mask, save_name

def main():
    cfg = parse_args()
    datamodule = get_dataset(cfg)
    model = get_model_with_config(cfg, datamodule)
    # model.load_state_dict(torch.load("saved/MDM_mask_1/checkpoints/epoch=1799_val_loss=0.29.ckpt")['state_dict'])
    model.load_state_dict(torch.load(cfg.MODEL.CHECKPOINT)['state_dict'])
    model = model.to(device)
    model.eval()

    text_encoder = AutoModel.from_pretrained('./deps/clip-vit-base-patch16').to(device)
    tokenizer = AutoTokenizer.from_pretrained('./deps/clip-vit-base-patch16')

    length = 50
    # text, sample, inpainting_mask, save_name = generate_taiji(length=length)
    # text, sample, inpainting_mask, save_name = generate_pelvis(length)
    text, sample, inpainting_mask, save_name = generate_sitcrying(length)

    with torch.no_grad():
        text_inputs = tokenizer(
            text, 
            padding="max_length",
            truncation=True,
            max_length=tokenizer.model_max_length,
            return_tensors="pt"
        )
        text_input_ids = text_inputs.input_ids
        caption = text_encoder.get_text_features(
            text_input_ids.to(text_encoder.device)
        ).cpu().numpy()

        smpls = model.inpainting({
            "text":[caption], 
            "length":[length],
            "sample":sample,
            "inpainting_mask":inpainting_mask
        }).cpu()
        # smpls = sample
        # smpls = smpls[~inpainting_mask.bool()]
        smpls = datamodule.recover_motion(smpls, local_only=False)
        np.save(save_name, smpls.v[..., [0,2,1]].numpy())

if __name__ == '__main__':
    main()