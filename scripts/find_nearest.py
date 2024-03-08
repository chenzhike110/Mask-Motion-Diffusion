import torch
from tqdm import tqdm
import numpy as np
from transformers import AutoModel, AutoTokenizer
from glob import glob

device = torch.device('cpu')
text_encoder = AutoModel.from_pretrained('./deps/clip-vit-base-patch16').to(device)
tokenizer = AutoTokenizer.from_pretrained('./deps/clip-vit-base-patch16')

with torch.no_grad():
    text = ["A person plays Tai Chi."]
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

min_f = 'datasets/HumanML3D/text_embeddings/014114.npy'
min_dist = 99999

# for f in tqdm(glob('datasets/HumanML3D/text_embeddings/*.npy')):
#     data = np.load(f)
#     dist = np.linalg.norm(data-caption, axis=-1)
#     if min(dist) < min_dist:
#         min_f = f
#         min_dist = min(dist)

print(min_f)

with open(min_f.replace('text_embeddings', 'texts').replace('.npy','.txt')) as f:
    print(f.readlines())



