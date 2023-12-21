import torch
from torch.utils.data import Dataset
from pycocotools.coco import COCO

<<<<<<< HEAD
class COCO_densepose(Dataset):
    """COCO densepose dataset with caption"""
    
annFile = "./datasets/COCO/annotations/captions_train2014.json"
coco_caps=COCO(annFile)

=======
class COCO(Dataset):
    """COCO densepose dataset with caption"""

annFile = "./datasets/COCO/annotations/captions_train2014.json"
coco_caps=COCO(annFile)
>>>>>>> 483f71c25ad588786ffd71f1d7bec3df7dc04878
