import torch
import numpy as np
from transformers import AutoModel, AutoTokenizer
from libs.config import parse_args
from libs.get_model import get_model_with_config, get_dataset

device = torch.device('cuda:1')

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
            ["a person walks forward, then sits on a seat."], 
            padding="max_length",
            truncation=True,
            max_length=tokenizer.model_max_length,
            return_tensors="pt"
        )
        text_input_ids = text_inputs.input_ids
        caption = text_encoder.get_text_features(
            text_input_ids.to(text_encoder.device)
        ).cpu().numpy()


    with torch.no_grad():
        smpls_ = model.sample({
            "text":[caption], 
            "length":[120]
        }).cpu()
        smpls, pose_body, pose_root, root_trans = datamodule.recover_motion(smpls_, local_only=False, normalized=cfg.MODEL.normalize, return_smpl=True)
        np.save('./demos/samples/walk_sits.npy', smpls.v.numpy())
        
        smpl_dict = {
            'pose': torch.cat([pose_root, pose_body, torch.zeros(pose_body.shape[0], 6)], dim=-1).numpy(),
            'trans': root_trans,
            'latents': smpls_
        }
        print(root_trans, pose_root)
        import joblib
        joblib.dump(smpl_dict, "demos/samples/walk_sits.pkl")


if __name__ == "__main__":
    main()

