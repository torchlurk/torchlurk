import torch
import torch.nn as nn
from pathlib import Path
import os
from torchvision.utils import save_image
from PIL import Image
from torchvision import transforms
from IPython.display import clear_output
from shutil import copyfile



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
                    
def clean_bw_imgs(path_to_imgs_dirs):
    """
    Cleans the tinyimagenet from its bw images.
    Args:
        path_to_imgs_dirs(str):path to the train or val folder of ImageNet
    """
    print("BW cleaning started")
    list_bw = []
    for i,(root, dirs, files) in enumerate(os.walk(path_to_imgs_dirs,topdown=False)):
        clear_output(wait=True)
        print("Progression:{:.2f}%".format(i/1000*100))
        for name in files:
            path = os.path.join(root, name)
            image = Image.open(path)
            image = transforms.ToTensor()(image)
            if image.shape != torch.Size([3,224,224]):
                list_bw.append(path)
                image = torch.stack([image,image,image],dim  =1).squeeze(0)
                assert(image.shape ==torch.Size([3,224,224]))
                save_image(image,path)
    print("BW files found:")
    for i in list_bw:
        print(i)
    print("BW cleaning terminated.")
                
def sample_imagenet(src_path_imgs,trgt_pathname_imgs,img_num_per_dir = 1):
    """
    create another directory similar to imagenet with a smaller number of images per class
    Args:
        src_path_imgs(str): path to train or val directory of ImageNet
        trgt_path_imgs(str): path + name of the folder which will keep the imagenet samples
    """
    assert(img_num_per_dir >=1)
    #target_path = "./data/exsmallimagenet"
    #src_path = "./data/tinyimagenet/train/"
    print("Start sampling")
    Path(trgt_pathname_imgs).mkdir(parents=True, exist_ok=True)
    for num_subfold,subfold in enumerate(os.listdir(src_path_imgs)):
        clear_output(wait=True)
        print("Progression:{:.2f}%".format(num_subfold/1000*100))
        subfold_trget_path = os.path.join(trgt_pathname_imgs,subfold)
        subfold_src_path = os.path.join(src_path_imgs,subfold)
        # create the directory
        Path(subfold_trget_path).mkdir(parents=True,exist_ok=True)
        for i,file in enumerate(os.listdir(subfold_src_path)):
            if i >= img_num_per_dir:
                break
            copyfile(os.path.join(subfold_src_path,file),os.path.join(subfold_trget_path,file))
    print("Sampling terminated.")