import os
import torch
import numpy as np
import codecs as cs
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

text_encoder = AutoModel.from_pretrained('./deps/clip-vit-base-patch16').cuda()
tokenizer = AutoTokenizer.from_pretrained('./deps/clip-vit-base-patch16')

dump_dir = "datasets/HumanML3D/text_embeddings"

with torch.no_grad():
    text_inputs = tokenizer(
        [""],
        padding="max_length",
        truncation=True,
        max_length=tokenizer.model_max_length,
        return_tensors="pt"
    )
    text_input_ids = text_inputs.input_ids
    caption = text_encoder.get_text_features(
        text_input_ids.to(text_encoder.device)
    ).cpu().numpy()

np.save(os.path.join(dump_dir, "-1.npy"), caption)

id_list = []
text_dir = "datasets/HumanML3D/texts/"
with cs.open(os.path.join("datasets/HumanML3D/all.txt"), 'r') as f:
    for line in tqdm(f.readlines()):
        with cs.open(os.path.join(text_dir, line.strip() + '.txt')) as fd:
            captions = []
            for line1 in fd.readlines():
                line_split = line1.strip().split('#')
                caption = line_split[0]
                captions.append(caption)
            text_inputs = tokenizer(
                captions,
                padding="max_length",
                truncation=True,
                max_length=tokenizer.model_max_length,
                return_tensors="pt"
            )
            text_input_ids = text_inputs.input_ids
            with torch.no_grad():
                captions = text_encoder.get_text_features(
                    text_input_ids.to(text_encoder.device)
                ).cpu().numpy()
            np.save(os.path.join(dump_dir, line.strip() + '.npy'), captions)
