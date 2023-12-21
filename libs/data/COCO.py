import torch
from torch.utils.data import Dataset
from pycocotools.coco import COCO

class COCO_densepose(Dataset):
    """COCO densepose dataset with caption"""
    
annFile = "./datasets/COCO/annotations/captions_train2014.json"
coco_caps=COCO(annFile)

