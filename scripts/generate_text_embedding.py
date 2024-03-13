import os
import torch
import numpy as np
import codecs as cs
from tqdm import tqdm
import clip

device = torch.device("cuda")

clip_model, clip_preprocess = clip.load('ViT-B/32', device='cuda', jit=False)
clip.model.convert_weights(clip_model) 
clip_model.eval()

dump_dir = "./data/HumanML/text_embeddings"

with torch.no_grad():
    text = clip.tokenize([''], truncate=True).to(device)
    feat_clip_text = clip_model.encode_text(text).float().cpu()

torch.save(feat_clip_text, os.path.join(dump_dir, "-1.pt"))

id_list = []
text_dir = "data/HumanML/texts/"
with cs.open(os.path.join("data/HumanML/all.txt"), 'r') as f:
    for line in tqdm(f.readlines()):
        with cs.open(os.path.join(text_dir, line.strip() + '.txt')) as fd:
            captions = []
            for line1 in fd.readlines():
                line_split = line1.strip().split('#')
                caption = line_split[0]
                captions.append(caption)
            text_inputs = text = clip.tokenize(captions, truncate=True).to(device)
            with torch.no_grad():
                captions = clip_model.encode_text(text_inputs).float().cpu()
            torch.save(captions, os.path.join(dump_dir, line.strip() + '.pt'))
