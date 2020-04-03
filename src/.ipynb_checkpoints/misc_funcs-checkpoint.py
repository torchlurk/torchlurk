import torch
import torch.nn as nn
from pathlib import Path
import os

def create_folders(path,direc_types,model_info):
    """
    create the directories to stock the images
    Args:
        origin_path(str): where to create these dirs
        direc_types(list:str): "gradients","max_activ","cropped"
        
    """
    Path(path).mkdir(parents=False, exist_ok=True)
    for dirtype in direc_types:
        subpath = os.path.join(path,dirtype)
        Path(subpath).mkdir(parents=False, exist_ok=True)
        for i,lay_info in enumerate(model_info):
            if (type(lay_info['lay']) == nn.Conv2d):
                subpath2 = os.path.join(subpath,lay_info['name'])
                Path(subpath2).mkdir(parents=False, exist_ok=True)
                for j,filt in enumerate(lay_info['filters']):
                    subpath3 = os.path.join(subpath2,str(filt["id"]))
                    Path(subpath3).mkdir(parents=False, exist_ok=True)