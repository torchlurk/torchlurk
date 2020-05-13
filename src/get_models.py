import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.transforms import ToTensor,ToPILImage
from torchvision.datasets import ImageFolder
from torchvision.utils import save_image
from torchvision import transforms
from PIL import Image

def get_alex_places():
    # load the pre-trained weights
    arch = 'alexnet'
    model_file = './models/%s_places365.pth.tar' % arch
    """
    if not os.access(model_file, os.W_OK):
        weight_url = 'http://places2.csail.mit.edu/models_places365/' + model_file
        os.system('wget ' + weight_url)
    """
    alex_places = models.__dict__[arch](num_classes=365)
    checkpoint = torch.load(model_file, map_location=lambda storage, loc: storage)
    state_dict = {str.replace(k,'module.',''): v for k,v in checkpoint['state_dict'].items()}
    alex_places.load_state_dict(state_dict)
    alex_places.eval()
    return alex_places