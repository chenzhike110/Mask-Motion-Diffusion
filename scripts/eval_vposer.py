import sys
sys.path.insert(0, './')

import torch
import numpy as np
from libs.models import VPoser

device = torch.device("cuda")
vposer = VPoser().to(device)

def covert_ckpt():
    pass

def show_sample(num_poses):
    import trimesh
    from libs.body_model import BodyModel
    body_fname = "./deps/body_models/smplh/neutral/model.npz"
    body = BodyModel(body_fname)

    poses = vposer.sample_poses(num_poses)['pose_body'].contiguous().view(num_poses, -1)
    with torch.no_grad():
        bodyM = body(pose_body=poses)

    scene = trimesh.Scene()
    meshes = []
    for i in range(num_poses):
        mesh = trimesh.Trimesh(vertices=bodyM.v[i].numpy(), faces=bodyM.f)
        scene.add_geometry(mesh)
    scene.show()

def eval_vae():
    # compute diversity
    from tqdm import tqdm
    from libs.metrics import calculate_frechet_distance, calculate_diversity_np
    batch_size = 128
    diversity_times = 300
    test_file = "./datasets/AMASS/test/pose_body.pt"
    test_file = torch.load(test_file).type(torch.float32)

    trgs = None
    for batch_idx in tqdm(range(len(test_file) // batch_size)):
        src = test_file[batch_idx*batch_size:(batch_idx+1)*batch_size, :].to(device)
        with torch.no_grad():
            trg = vposer(src)['pose_body'].contiguous().view(src.shape[0], -1).cpu().numpy()
            if trgs is None:
                trgs = trg
            else:
                trgs = np.concatenate((trgs, trg))

    diversity_times = min(test_file.shape[0], diversity_times)
    
    test_file = test_file.numpy()
    mu1 = np.mean(test_file)
    cov1 = np.cov(test_file, rowvar=False)
    mu2 = np.mean(trgs)
    cov2 = np.cov(trgs, rowvar=False)

    FID = calculate_frechet_distance(mu1, cov1, mu2, cov2)
    DIV = calculate_diversity_np(trgs, diversity_times)
    GT_DIV = calculate_diversity_np(test_file, diversity_times)

    print("Model FID score: ", FID)
    print("Model DIV score: ", DIV)
    print("GT DIV score: ", GT_DIV)

if __name__ == "__main__":
    # checkpoint = torch.load("./saved/Vposer/best_model.ckpt", map_location=device)
    checkpoint = torch.load("./saved/Vposer/checkpoints/epoch=1.ckpt", map_location=device)
    vposer.load_state_dict(checkpoint['state_dict'])
    vposer.eval()
    eval_vae()




