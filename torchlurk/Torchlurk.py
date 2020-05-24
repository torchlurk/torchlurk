from IPython.core.debugger import set_trace
from IPython.display import clear_output

import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.transforms import ToTensor,ToPILImage
from torchvision.datasets import ImageFolder
from torchvision.utils import save_image
from torchvision import transforms
from PIL import Image
from time import perf_counter 
from copy import deepcopy
import matplotlib.pyplot as plt
from shutil import copyfile
from enum import Enum

import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
import pickle
from copy import deepcopy
from pathlib import Path
from collections import OrderedDict
from collections import Counter
import multiprocessing


import dill
from .ImageFolderWithPaths import ImageFolderWithPaths
from .Projector import Projector

from .app import start as app_start
from .misc_funcs import create_folders
#libraries
from .lib.pytorch_cnn_visualizations.cnn_layer_visualization import CNNLayerVisualization
from .lib.pytorch_cnn_visualizations.layer_activation_with_guided_backprop import GuidedBackprop
from .lib.pytorch_cnn_visualizations.misc_functions import save_gradient_images

class State(Enum):
    idle=1
    compute_top=2
    compute_activ=3
    compute_grad=4


class Lurk():
    """
    Lurker class: one lurker can be instanciated per trained pytorch network. Several methods allow to generate various types of data concerning the network and can be visualized thanks to the bash command TOCOMPLETE

    Attributes:
        model (torch module): model to compute a lurker for
        preprocess (nn.Sequential): preprocessing used when training the model
        state (State): state for live update
        GEN_IMGS_DIR (str): directory where to save the generated images. The lurker will create various subdirectories in it.
        IMGS_DIR (str): directory where to load the training images
        JSON_PATH_WRITE (str): directory where to save the json
        NUMB_PATH (str): path to a numb image

        N_TOP_AVG (int): number of avg spikes images per filter
        N_TOP_MAX (int): number of max spikes images per filter
        side_size (int): size in pixel of a typical training sample

        dataset (torch.dataset): dataset constructed to compute the top images
        ImageFolderWithPaths (torch.imagefolder): imageloader constructed to compute the top images
        CLASS2LAB (dic): convert the class (ex:"rabbit") to its label (ex:256)
        LAB2CLASS (dic): inverse of CLASS2LAB
        TITLE2CLASS (dic): converts the title found in the filename ("n01243532") to the class name ("rabbit")
        title_counts: keeps track of the filters affinity for the classes in terms of avg/max activations.
    """
    def __init__(self,model,preprocess,
                 save_gen_imgs_dir,save_json_path,imgs_src_dir,
                 n_top_avg=3,n_top_max=3,load_json_path=None,side_size=224,title2class=None):
        """
        **Constructor**:
        
        Args:
            model (torch module): model to compute a lurker for            the trained model
            preprocess (nn.Sequential): preprocessing used when training the model
            save_gen_imgs_dir (str): directory where to save the generated images. The lurker will create various subdirectories in it.
            save_json_path (str): directory where to save the json
            imgs_src_dir (str): directory where to load the training images
            n_top_avg (int): number of avg spikes images per filter
            n_top_max (int): number of max spikes images per filter
            load_json_path (str): path to a previously computed json file: NB the other arguments need to be similar as during this first run. If set to a value,
            the lurker will load the previously computed informations, if set to None, will start a new lurker from scratch
            
            title2class (dic): converts the title found in the filename ("n01243532") to the class name ("rabbit")
        """
        
        ##################### TODO: get rid of dev#####################
        # allow to run reduced computations
        self.DEVELOPMENT = False
        #number of layers we compute stuff for in development mode
        self.N_LAYERS_DEV = 1
        #number of filters we compute stuff for in development mode
        self.N_FILTERS_DEV = 1
        ###############################################################

        #model to compute a lurker for
        self.model = model
        #preprocessing used when training the model
        self.preprocess = preprocess
        self.side_size = side_size
        #state of the lurker: takes values in "idle","compute top","compute activ","compute grad"
        self.state = State.idle

        #directory where to save the generated images
        self.GEN_IMGS_DIR = save_gen_imgs_dir
        #directory where to load the training images
        self.IMGS_DIR = imgs_src_dir
        
        #where to save the json
        self.JSON_PATH_WRITE = save_json_path
        #number of avg spikes images per filter
        self.N_TOP_AVG = n_top_avg
        # number of max spikes images per filter
        self.N_TOP_MAX = n_top_max
        
        #path to the numb image
        self.NUMB_PATH = "data/numb.png"
        
        self.dataset = ImageFolderWithPaths(self.IMGS_DIR,transform=self.preprocess)
        self.data_loader = torch.utils.data.DataLoader(self.dataset, batch_size=1, shuffle=True)
        # each class has 3 kinds of representation: (1)class titles (ex: "penguin") (2) dirname (ex:"n02018795") (3) label (ex:724)
        self.CLASS2LAB = self.dataset.class_to_idx
        self.LAB2CLASS = {i:j for j,i in self.CLASS2LAB.items()}
        self.TITLE2CLASS = title2class
        
        if load_json_path is not None:
            # loading information from a previously computed json file
            self.load_from_json(load_json_path)
        else:
            # building the information from scratch
            self.__build_model_info()
        self.save_to_json()
        
        create_folders(self.GEN_IMGS_DIR,["avg_act","avg_act_grad","max_act","max_act_grad","max_act_cropped","max_act_cropped_grad","filt_viz"],self.model_info)
        
        
        self.title_counts = dict(zip(self.LAB2CLASS.values(),[0] *len(self.LAB2CLASS)))
        #initiate the number of counts for the classes
        self.__init_class_counts()
    
    
    ################################ Building/Loading ################################
    
    def __build_model_info(self):
        """
        build the model_info from scratch: the model_info is the main data structure to store all the information about computed data, loading paths serving and states.
        """
        model_info = []
        layers = []
        #construct the data structure
        for layer in list(self.model.features.named_children()):
            lay_info = {'id':layer[0],
                      'lay':layer[1],
                      'name':str(layer[1]).split('(')[0] + "_" + str(layer[0]) 
                    }
            if (isinstance(layer[1],(nn.Conv2d,nn.MaxPool2d))):
                layers.append(layer[1])
            if (isinstance(lay_info['lay'],nn.Conv2d)):     
                n_input = lay_info['lay'].in_channels
                n_output = lay_info['lay'].out_channels
                lay_info['n_input'] = n_input
                lay_info['n_output'] = n_output
                lay_info['deproj'] = Projector(deepcopy(layers),224)
                lay_info["filters"] = []
                for i in range(n_output):
                    lay_info["filters"].append({
                        "id":i,
                        "avg_spikes":[0 for i in range(self.N_TOP_AVG)],
                        "avg_imgs":[self.NUMB_PATH for i in range(self.N_TOP_AVG)],
                        "avg_imgs_grad":[self.NUMB_PATH for i in range(self.N_TOP_AVG)],
                        "max_spikes":[0 for i in range(self.N_TOP_MAX)],
                        "max_slices":[[[0,0],[0,0]]for i in range(self.N_TOP_MAX)],
                        "max_imgs":[self.NUMB_PATH for i in range(self.N_TOP_MAX)],
                        "max_imgs_crop":[self.NUMB_PATH for i in range(self.N_TOP_MAX)],
                        "max_imgs_grad":[self.NUMB_PATH for i in range(self.N_TOP_MAX)],
                        "max_imgs_crop_grad":[self.NUMB_PATH for i in range(self.N_TOP_MAX)],
                        "filter_viz":self.NUMB_PATH,
                        "histo_counts_max":OrderedDict(zip(self.LAB2CLASS.values(),[0] *len(self.LAB2CLASS))),
                        "histo_counts_avg":OrderedDict(zip(self.LAB2CLASS.values(),[0] *len(self.LAB2CLASS)))
                    })
            elif (type(lay_info['lay']) == nn.Linear):
                    n_input = lay_info['lay'].in_features
                    n_output = lay_info['lay'].out_features
                    lay_info['n_output'] = n_output
                    #lay_info["filters"] = [empty_filter.copy() for i in range(n_output)]
            model_info.append(lay_info)
            self.model_info = model_info
            self.conv_layinfos = [lay_info for lay_info in self.model_info if isinstance(lay_info['lay'],nn.Conv2d)]
            
    def __set_state(self,state):
        """ set the state to the value passed in parameter"""
        self.state = state
        with open(self.JSON_PATH_WRITE, 'r+') as f:
            data = json.load(f)
            data['state'] = self.state.name
            f.seek(0)        
            json.dump(data, f, indent=2)
            f.truncate()

    def save_to_json(self,alternate_json_path=None):
        """
        Save the json information.
        
        Args:
            alternate_json_path (str): Alternate path to save the json. By default, the lurker will save to JSON_PATH_WRITE.
        """
        path_write = self.JSON_PATH_WRITE if alternate_json_path is None else alternate_json_path
        obj = {'current_json':str(Path("/").joinpath(Path(self.JSON_PATH_WRITE)))}
        with open("saved_model/.current.json", 'w') as f:
            json.dump(obj, f, indent = 2)
        model_info2 = deepcopy(self.model_info)
        for lay_info in model_info2:
            if (isinstance(lay_info['lay'],nn.Conv2d)):
                del lay_info['deproj']
            del lay_info['lay']
        with open(path_write, 'w') as fout:
            model_info2 = {'state':self.state.name,'infos':model_info2}
            json.dump(model_info2, fout, indent = 2)
        print("json saving done!")
        
    def load_from_json(self,load_path):
        """
        loads a lurker from a previously computed json. **Watch out: any computation made after loading will overwrite the json file passed in parameter. Please backup the file if you don't wish for any modification.**
        
        Args:
            load_path (str): path to the json to load a lurker from
        """
        layers = []
        with open(load_path, 'r') as fin:
            model_info = json.load(fin)['infos']
        for lay_info,layer in zip(model_info,self.model.features):
            lay_info['lay'] = layer
            if (isinstance(layer,(nn.Conv2d,nn.MaxPool2d))):
                layers.append(layer)
            if (isinstance(layer,nn.Conv2d)):
                lay_info['deproj'] = Projector(deepcopy(layers),224)
        self.model_info = model_info
        self.conv_layinfos = [lay_info for lay_info in self.model_info if isinstance(lay_info['lay'],nn.Conv2d)]
        self.__check_imgs_exist()
        print("Loading from json done!") 
    
        
    def save_to_dill(self,path):
        """
        save to dill format: similar to pickle format,but handle additional data formats
        """
        torch.save(self,path, pickle_module=dill)
        print("dill saving done!")
        
        

    @staticmethod
    def load_from_dill(load_path,overwrite=False,alternate_json_path=None):
        """
        load the lurker from a dill (pickle-like) file
        
        Args:
            load_path(str): path to the dill file
            overwrite (Bool): whether to overwrite the JSON the lurker reads from
            alternate_path (str):  specific path to write the next json. By default, adds a "_copy" extension to the json path set in the pickled instance. Can have a value assigned only if overwrite is set to False.
        """
        assert(not (overwrite and alternate_json_path is not None))
        lurker = torch.load(load_path, pickle_module=dill)
        if not overwrite:
            if alternate_json_path is None:
                path = Path(lurker.JSON_PATH_WRITE)
                name = path.stem + "_copy"
                new_path = str(path.with_name(name).with_suffix(".json"))
                lurker.JSON_PATH_WRITE = new_path
                copyfile(path,new_path)
                
            else:
                path = Path(alternate_json_path)
                assert(path.is_file() and path.suffix == ".json")
                lurker.JSON_PATH_WRITE = alternate_json_path
        lurker.__check_imgs_exist()
        print("Loading from dill done!") 
        return lurker    
    
    def __check_imgs_exist(self): 
        """
        check that all the images in the json file indeed exist and are at the right position when loading
        """
        try:
            type_keys = ["avg_imgs","avg_imgs_grad","max_imgs","max_imgs_grad"]
            for lay_info in self.conv_layinfos:
                for filtr in lay_info['filters']:
                    for key in type_keys:
                        for el in filtr[key]:
                            if not Path(el).exists():
                                raise FileNotFoundError(el)
                    if not Path(filtr["filter_viz"]).exists():
                        raise FileNotFoundError(filtr["filter_viz"])
        except FileNotFoundError as e:
            print(key)
            print("Non coherent path in the loaded object:{}".format(e))
            
    
    ################################ Practical ################################

            
    def __get_filt_dir(self,dir_type,layer_name,filter_id):
        """
        return the path to the appropriate folder
        
        Parameters
        ----------
        dir_type : str
            One of "avg_act","avg_act_grad","max_act","max_act_grad","max_act_cropped","max_act_cropped_grad","filt_viz"
        layer_name : str
            name of the layer
        filter_id : int/str
            index of the filter
        """
        return os.path.join(self.GEN_IMGS_DIR,dir_type,str(layer_name),str(filter_id))
    
   
    
    ################################################# Generated images #######################################################

    ################################ average/maximum activation images ################################
    def compute_top_imgs(self,verbose = False,compute_max=True,compute_avg=True,save_loc=True):
        """
        compute the average and max images for all the layers of the model_info such that each filter of each layer knows what are
        its favourite images (write down the link to the avg/max images in the json)
        
        Args:
            compute_max (bool): whether to compute the top maximum activation images
            compute_avg (bool): whether to compute the top average activation images
            save_loc (bool): save the images to the GEN_IMGS_DIR from the SRC_DIR (training set) upon completion of the function call
        """
        self.__set_state(State.compute_top)
        assert(compute_max or compute_avg)
        self.__reset_histos()
        for j,(datas,labels,paths) in enumerate(self.data_loader):
            print("Progression update favimgs:{:.2f} %".format(j/len(self.data_loader) * 100))
            for i,lay_info in enumerate(self.model_info):
                clear_output(wait=True)
                if verbose:
                    print("AvgMax update:{}/{}:{:.2f} %..".format(i,len(self.model_info),100*j/ len(data_loader)))

                #datas: Batchsize x Numberfilter x Nout x Nout
                datas = lay_info['lay'](datas)
                if (not isinstance(lay_info['lay'],nn.Conv2d) ):
                    continue
                if (i >=self.N_LAYERS_DEV and self.DEVELOPMENT):
                    break

                batch_size = datas.size(0)
                filter_nb = datas.size(1)
                width = datas.size(3)

                #spikes: Batchsize x Filternumber
                max_spikes,max_pos = datas.view([batch_size,filter_nb,-1]).max(dim = 2)
                max_rows = max_pos / width
                max_cols = max_pos % width

                avg_spikes = datas.view([batch_size,filter_nb,-1]).mean(dim = 2)
                if compute_max:
                    self.__update_filters_max_imgs(lay_info,max_spikes,paths,max_rows,max_cols,labels)
                if compute_avg:
                    self.__update_filters_avg_imgs(lay_info,avg_spikes,paths,labels)
                #save the whole model
                if (j > 200 and self.server is not None):
                    #save the information for live udpate
                    self.save_to_json()

        self.__normalize_histos()
        self.__sort_filters_spikes()
        if save_loc:
            self.save_avgmax_imgs()
        self.__save_cropped()
        self.__set_state(State.idle)
        self.save_to_json()

        
    def save_avgmax_imgs(self):
        """
        save the top average/maximum activation images figuring in the json to the GEN_IMGS_DIR directory. This allows the lurker to become independant of the training set for subsequent visualization/computations.
        """
        for i,lay_info in enumerate(self.conv_layinfos):
            clear_output(wait=True)
            print("Saving top_avg_max progression {:.2f} %".format(i/len(self.conv_layinfos)*100))
            for filtr in lay_info['filters']:
                for agg in ["avg","max"]:
                    for i,src_path in enumerate(filtr["{}_imgs".format(agg)]):
                        new_dir = Path(self.__get_filt_dir("{}_act".format(agg),lay_info["name"],filtr["id"]))
                        name = Path(src_path).name
                        new_path= new_dir.joinpath(name)
                        #we open the path pointing to the training set
                        im = Image.open(src_path)
                        im.save(new_path)
                        filtr["{}_imgs".format(agg)][i] = str(new_path)
        self.save_to_json()
                    
    def __sort_filters_spikes(self):
        """
        sorts the spikes and respective paths of the filters inplace
        """
        for lay_info in self.conv_layinfos:
            for filtr in lay_info['filters']:
                max_indx = np.argsort(filtr["max_spikes"])[::-1]
                filtr["max_spikes"] = np.array(filtr["max_spikes"])[max_indx].tolist()
                filtr["max_imgs"] = np.array(filtr["max_imgs"])[max_indx].tolist()
                filtr["max_slices"] = np.array(filtr["max_slices"])[max_indx].tolist()

                avg_indx = np.argsort(filtr["avg_spikes"])[::-1]
                filtr["avg_spikes"] = np.array(filtr["avg_spikes"])[avg_indx].tolist()
                filtr["avg_imgs"] = np.array(filtr["avg_imgs"])[avg_indx].tolist()
    

    def __update_filters_max_imgs(self,lay_info,batch_spikes,paths,max_rows,max_cols,labels):
        #as many spikes in batch_spikes as there are samples in batch
        for spikes,path,label,rows,cols in zip(batch_spikes,paths,labels,max_rows,max_cols):
            #at this stage there are as many spike in spikes as there are filters
            for k,(filt,spike,row,col) in enumerate(zip(lay_info["filters"],spikes.detach().numpy(),rows,cols)):
                #compute the histogram with maximal values
                filt["histo_counts_max"][self.LAB2CLASS[label.item()]] += float(spike)
                #compute the minimum spike for the filter
                min_indx = np.argmin(filt["max_spikes"])
                min_spike = min(filt["max_spikes"])
                
                if (spike > min_spike and not (path in filt["max_imgs"])):
                    ((x1,x2),(y1,y2)) = lay_info["deproj"].chain(((row.item(),row.item()),(col.item(),col.item())))
                    assert(isinstance(x1,int) and isinstance(x2,int) and isinstance(y1,int) and isinstance(y2,int))
                    filt["max_slices"][min_indx] = ((x1,x2),(y1,y2))
                    filt["max_imgs"][min_indx] = path
                    filt["max_spikes"][min_indx] = float(spike)
                    
    def __update_filters_avg_imgs(self,lay_info,batch_spikes,paths,labels):
        #as many spikes in batch_spikes as there are samples in batch
        for spikes,path,label in zip(batch_spikes,paths,labels):
            #at this stage there are as many spike in spikes as there are filters
            for k,(filt,spike) in enumerate(zip(lay_info["filters"],spikes.detach().numpy())):
                #compute the histogram with avg values
                filt["histo_counts_avg"][self.LAB2CLASS[label.item()]] += float(spike)
                #compute the minimum spike for the filter
                min_indx = np.argmin(filt["avg_spikes"])
                min_spike = min(filt["avg_spikes"])
                if (spike > min_spike and not (path in filt["avg_imgs"])):
                    filt["avg_imgs"][min_indx] = path
                    filt["avg_spikes"][min_indx] = float(spike)
                

    def __save_cropped(self,grad = False,verbose=False):
        """
        iterate on the model_info to save a cropped version of the images
        Args:
            grad(Bool): whether to save the gradients versions
        """
        ext = "_grad" if grad else ""
        filtrlist = "max_imgs{}".format(ext)
        folder = "max_act_cropped{}".format(ext)
        filtrtargetlist = "max_imgs_crop{}".format(ext)

        for i,lay_info in enumerate(self.conv_layinfos):
            clear_output(wait=True)
            if verbose:
                print("Progression:{} %".format(i/len(conv_layinfos)*100))
            for filtr in lay_info['filters']:
                for i,(src_path,slices) in enumerate(zip(filtr[filtrlist],filtr['max_slices'])):
                    if (src_path == self.NUMB_PATH):
                        continue
                    src_path = Path(src_path)
                    target_dir = Path(self.__get_filt_dir(folder,lay_info['name'],filtr['id']))
                    file_name = src_path.name
                    target_path = target_dir.joinpath(file_name)
                    
                    image = ToTensor()(Image.open(src_path))
                    ((x1,x2),(y1,y2)) = slices
                    cropped = image[:,x1:x2+1,y1:y2+1]
                    ToPILImage()(cropped).save(str(target_path))
                    filtr[filtrtargetlist][i] = str(target_path)
                    
    ################################ Histograms related ################################

            
            
    def __init_class_counts(self,):
        """
        create the dictionary which counts the number of images per classes in the dataset
        """
        for subdir in Path(self.IMGS_DIR).iterdir():
            if subdir.is_dir():
                count = len(list(subdir.iterdir()))
                title = subdir.name
                self.title_counts[title] += count
            
    def __reset_histos(self):
        """
        reset the counts for the histograms counts
        """
        for lay_info in self.conv_layinfos:
            for filt in lay_info['filters']:
                filt['histo_counts_max'] = dict(zip(self.LAB2CLASS.values(),[0] *len(self.LAB2CLASS)))
                filt['histo_counts_avg'] = dict(zip(self.LAB2CLASS.values(),[0] *len(self.LAB2CLASS)))
            
    def __normalize_histos(self):
        """
        average the counts of the histograms wrt to the number of samples in the classes of the dataset
        """
        for lay_info in self.conv_layinfos:
            for filt in lay_info['filters']:
                for key in filt['histo_counts_max'].keys():
                    filt['histo_counts_max'][key] /= self.title_counts[key]
                    filt['histo_counts_max'][key] = max(filt['histo_counts_max'][key],0)
                for key in filt['histo_counts_avg'].keys():
                    filt['histo_counts_avg'][key] /= self.title_counts[key]
                    filt['histo_counts_avg'][key] = max(filt['histo_counts_avg'][key],0)
                filt['histo_counts_max'] = OrderedDict(sorted(filt['histo_counts_max'].items(),key = lambda x: x[1])[::-1])
                filt['histo_counts_avg'] =  OrderedDict(sorted(filt['histo_counts_avg'].items(),key = lambda x: x[1])[::-1])
                

                    
    ################################ Layer visualizations ################################
                    
    def compute_viz(self,num_imgs_per_layer=None,ratio_imgs_per_layer=None,first_n_imgs=None):
        """
        Compute the filter visualization for all layers **only one of num_imgs_per_layer,ratio_imgs_per_layer or first_n_imgs can be have a value assigned.** By default, compute the visualisation for all filters of all layers (can be computationally demanding)
        
        Args:
            num_imgs_per_layer (int): number of filters to compute the visualization for for each class
            ratio_imgs_per_layer(float): ratio of filters to compute the visualization for for each class
            
        .. note::
            
            The lurker will first check if any file exists at its write locations for the filter visualisations/ gradients. If this is the case, it will update its json path accordingly, but won't compute any new images for computational reasons. Please remove any existing images you wish to re-compute filter visualisations/ gradients for.
            
        """
        self.__set_state(State.compute_activ)
        checks = [num_imgs_per_layer is not None , ratio_imgs_per_layer is not None,first_n_imgs is not None]
        assert(sum(checks)==1 or sum(checks)==0)
        for lay_indx,lay_info in enumerate(self.conv_layinfos):
            print("Layer {}:".format(lay_info['id']))
            N = lay_info["lay"].out_channels
           
            if checks[0]:
                indexes = np.arange(N)
                np.random.shuffle(indexes)
                indexes = indexes[:num_imgs_per_layer]
            elif checks[1]:
                indexes = np.arange(N)
                np.random.shuffle(indexes)
                lim = int(ratio_imgs_per_layer * N)
                indexes = indexes[:lim]
            elif checks[2]:
                indexes = np.arange(N)[:first_n_imgs]
            self.compute_layer_viz(int(lay_info['id']),indexes)
        self.__set_state(State.idle)


    def compute_layer_viz(self,layer_indx,filter_indexes = None):
        """
        compute  and save the filter visualisation as an image. Compute it only for filters for which
        it has not been computed yet: **you need to delete the existing images manually if you wish for a refresh**.
        
        Args:
            layer_indx (int): layer to compute the filter for
            filter_indexes (list of int):  indexes of filters to compute the visualizations for. By default, compute it for each filter.
            
        .. note::
            
            The lurker will first check if any file exists at its write locations for the filter visualisations/ gradients. If this is the case, it will update its json path accordingly, but won't compute any new images for computational reasons. Please remove any existing images you wish to re-compute filter visualisations/ gradients for.
        """
        self.__set_state(State.compute_activ)
        lay_info = self.conv_layinfos[layer_indx]
        if filter_indexes is None:
            filter_indexes = range(lay_info["lay"].out_channels)
        else:
            for i in filter_indexes:
                if (i >= lay_info["lay"].out_channels or i<0):
                    raise IndexError("filter_indexes must have lower value than layer output number.")
        layer_name = lay_info["name"]
        pre_existing = []
        for filt in lay_info["filters"]:
            name = "{}_{}_max_activ.jpg".format(lay_info['name'],filt['id'])
            filt_path = self.__get_filt_dir("filt_viz",lay_info["name"],filt["id"])
            filt_path = os.path.join(filt_path,name)
            try:
                f = open(filt_path)
                filt["filter_viz"] = filt_path
                pre_existing.append(filt["id"])
                f.close()
            except FileNotFoundError:
                pass

        filter_indexes = [i for i in filter_indexes if not i in pre_existing]
        print("Effectively computing the filters:"filter_indexes)

        for j,filt_indx in enumerate(filter_indexes):
            print("Filter {} / {}".format(j+1,len(filter_indexes)))
            filt = lay_info['filters'][filt_indx]
            visualizer = CNNLayerVisualization(self.model.features, 
                                               selected_layer=int(lay_info['id']), 
                                               selected_filter=filt_indx,
                                               side_size=self.side_size)
            act_max_img = visualizer.visualise_layer_with_hooks()
            filt_path = self.__get_filt_dir("filt_viz",lay_info['name'],filt['id'])
            name = "{}_{}_max_activ.jpg".format(lay_info['name'],filt['id'])
            filt_path = os.path.join(filt_path,name)
            #save the image
            ToPILImage()(act_max_img).save(filt_path)
            print("Vis saved!")
            filt["filter_viz"] = filt_path
            self.save_to_json()
        print("Visualization done!")
        self.__set_state(State.idle)


        
    ################################ Gradients ################################
    def compute_grads_layer(self,lay_indx,compute_avg = True,compute_max = True,first_n_imgs=None):
        """
        compute the gradients for the top avg and/or max images of the layer corresponding to lay_indx.
        
        Args:
            layer_indx (int): index of the layer
            compute_avg (bool): whether to compute the gradient for the top average
            compute_max (bool): whether to compute the gradient for the top max
            first_n_imgs (int): if initialized, compute only the gradient for the first first_n_imgs filters of the layer
            
        .. note::
            
            The lurker will first check if any file exists at its write locations for the filter visualisations/ gradients. If this is the case, it will update its json path accordingly, but won't compute any new images for computational reasons. Please remove any existing images you wish to re-compute filter visualisations/ gradients for.
        """
        self.__set_state(State.compute_grad)
        gbp = GuidedBackprop(self.model)
        lay_info = self.conv_layinfos[lay_indx]
        for j,filt in enumerate(lay_info['filters']):
            clear_output(wait=True)
            if (first_n_imgs is not None):
                print("Grads Progression:layer{} {:.2f}%".format(lay_indx,j/first_n_imgs*100))
                if (j > first_n_imgs):
                    break
            else:
                print("Grads Progression:layer{} {:.2f}%".format(lay_indx,j/len(lay_info[lay_indx]['filters'])*100))
            if compute_avg:
                path = self.__get_filt_dir("avg_act_grad",lay_info['name'],filt["id"])
                self.__compute_grads_filt(gbp,filt,path,lay_info['id'],"avg_imgs")
            if compute_max:
                path = self.__get_filt_dir("max_act_grad",lay_info['name'],filt["id"])
                self.__compute_grads_filt(gbp,filt,path,lay_info['id'],"max_imgs")
                self.__save_cropped(grad= True)
        self.save_to_json()
        self.__set_state(State.idle)

    def compute_grads(self,compute_avg = True,compute_max = True,first_n_imgs=None,verbose = False):
        """
        compute the gradients for the fav images of all filters of all layers for the model_info
        
        Args:
            model_info (dic): as described above
            origin_path (str): path where to store the folders containing the gradient images
            compute_avg (bool): whether to compute the gradient for the top average
            compute_max (bool): whether to compute the gradient for the top max
            first_n_imgs (int): if initialized, compute only the gradient for the first first_n_imgs filters of **each layer**
            verbose (bool): displays progression info
            
        .. note::
            
            The lurker will first check if any file exists at its write locations for the filter visualisations/ gradients. If this is the case, it will update its json path accordingly, but won't compute any new images for computational reasons. Please remove any existing images you wish to re-compute filter visualisations/ gradients for.
            
        """
        self.__set_state(State.compute_grad)
        gbp = GuidedBackprop(self.model)
        for i,lay_info in enumerate(self.conv_layinfos):
            if (i >= self.N_LAYERS_DEV and self.DEVELOPMENT):
                break
            for j,filt in enumerate(lay_info['filters']):
                clear_output(wait=True)
                if (self.DEVELOPMENT):
                    print("Grads Progression:layer{}/{} {:.2f}%".format(i+1,self.N_LAYERS_DEV,j/self.N_FILTERS_DEV*100))
                    if (j >=self.N_FILTERS_DEV):
                        break
                else:
                    if (first_n_imgs is not None):
                        print("Grads Progression:layer{}/{} {:.2f}%".format(i+1,len(self.conv_layinfos),j/first_n_imgs*100))
                        if (j > first_n_imgs):
                            break
                if compute_avg:
                    path = self.__get_filt_dir("avg_act_grad",lay_info['name'],filt["id"])
                    self.__compute_grads_filt(gbp,filt,path,lay_info['id'],"avg_imgs")
                if compute_max:
                    path = self.__get_filt_dir("max_act_grad",lay_info['name'],filt["id"])
                    self.__compute_grads_filt(gbp,filt,path,lay_info['id'],"max_imgs")
                    self.__save_cropped(grad= True)
                #self.save_to_json()
        self.__set_state(State.idle)

    def __compute_grads_filt(self,gbp,filt,path,lay_id,img_type):
        """
        compute the gradients wrt to the favourite images of a filter filt.
        
        Args:
            gbp (GuidedBackprop): fitted on the model
            filt (dic): filter from a layer
            path (str): path to the folder where to store the gradient images
            img_type (str): either "avg_imgs" or "max_imgs"
        """
        grad_strindx = "avg_imgs_grad" if img_type == "avg_imgs" else "max_imgs_grad"

        for i,img_path in enumerate(filt[img_type]):
            if (img_path == self.NUMB_PATH or Path(img_path).stem == "numb"):
                continue   

            #name of the image
            img_name = Path(img_path).stem + "_grad.jpg"
            #joined path and imagename
            grad_path = Path(path).joinpath(img_name)

            #if the image already exists, we don't compute it again
            if grad_path.exists():
                print("grad_exist!")
                filt[grad_strindx][i] = str(grad_path)
                continue
            image = Image.open(img_path)
            image = self.preprocess(image).unsqueeze(0)
            image.requires_grad = True
            img_path = Path(img_path)

            class_name = self.TITLE2CLASS[img_path.name.split("_")[0]]
            gradient = gbp.generate_gradients(image,self.CLASS2LAB[class_name],lay_id,filt['id'])
            #normalization of the gradient
            gradient = gradient - gradient.min()
            gradient /= gradient.max()
            im = ToPILImage()(gradient[0])
            im.save(grad_path)
            filt[grad_strindx][i] = str(grad_path)
            
    ################################ Plotting ################################
    def __repr__(self):
        return self.model.__repr__()
    
    def plot_hist(self,layer_indx,filt_indx,hist_type="max",num_classes=30):
        """
        plot the sorted average values of avg/max spikes over the dataset for each class as an histogram
        
        Args:
            layer_indx (int): index of the layer
            filt_indx (int): index of the filter
            hist_type (str): either "max" or "avg": which hist to plot
        """
        assert(hist_type == "max" or hist_type == "avg")
        lay_info = self.model_info[layer_indx]
        assert(isinstance(lay_info['lay'],nn.Conv2d))
        filt = lay_info['filters'][filt_indx]
        obj = filt['histo_counts_max'] if hist_type == "max" else filt['histo_counts_avg']
        fig,ax = plt.subplots(1,1,figsize=(20,3))
        keys = [i for i in list(obj.keys())[::-1]][:num_classes]
        values = [i for i in list(obj.values())][:num_classes]
        ax.xaxis.set_tick_params(rotation=45)
        ax.bar(keys,values,color="#ee4c2c")

    def plot_filter_viz(self,layer_indx,filt_indx):
        """
        plot the filter visualization for the corresponding layer/filter combination
        
        Args:
            layer_indx (int): index of the layer
            filt_indx (int): index of the filter
        """
        filt = self.conv_layinfos[layer_indx]['filters'][filt_indx]
        im = Image.open(filt["filter_viz"])
        plt.imshow(np.asarray(im))
        plt.axis('off')
        plt.savefig("filt_viz.png")
        

    def plot_top(self,kind,layer_indx,filt_indx,plot_imgs=True,plot_grad=False):
        """
        plot the top activation samples of the train set for a given filter
        
        Args:
            kind (str): either "avg" or "max"
            layer_indx (int): index of the layer
            filt_indx (int): index of the filter
            plot_imgs (bool): True to plot the images
            plot_grad (bool): True to plot the gradients
            
        .. note::
            
            Either one of plot_imgs or plot_grad must be set to True.
            
        """
        filt = self.conv_layinfos[layer_indx]['filters'][filt_indx]
        imgs = []
        grads = []
        if (not plot_grad and not plot_imgs):
            raise NameError("No data to display: set one of plot_grad or polt_imgs to True.")
        if plot_imgs:
            imgs = filt['{}_imgs'.format(kind)]
        if plot_grad:
            grads = filt["{}_imgs_grad".format(kind)]
        N = max(len(imgs),len(grads))
        M = 2 if sum([plot_imgs,plot_grad])== 2 else 1
        fig,axes = plt.subplots(M,N,figsize=(20,5))
        imgs = imgs + grads
        for i,(ax,img) in enumerate(zip(axes.flatten(),imgs)):
            im_pil = Image.open(img)
            ax.axis('off')
            ax.imshow(np.asarray(im_pil),interpolation='nearest')
            
    def plot_crop(self,layer_indx,filt_indx,grad=False):
        """
        plot the top max images along with the cropped version which leads to the highest activating output for the given filter
        
        Args:
            layer_indx (int): index of the layer
            filt_indx (int): index of the filter
            grad (bool): whether to plot grad or the original image
            grad: whether to plot the grad equivalent 
        """ 
        key = "max_imgs"
        key_crop = "max_imgs_crop" 
        if grad:
            key += "_grad"
            key_crop += "_grad"
        filt = self.conv_layinfos[layer_indx]['filters'][filt_indx]
        imgs = filt[key]
        crop_imgs = filt[key_crop]
        fig,axes = plt.subplots(2,len(imgs),figsize=(20,4.5))
        for i,(ax,img) in enumerate(zip(axes.flatten(),imgs+crop_imgs)):
            ax.axis('off')
            im_pil = Image.open(img)
            ax.imshow(np.asarray(im_pil))
            
    ################################ Serving ################################
    def serve(self,port=5000):
        """
        serves the application on a localport
        
        Args:
            port (int): the port to serve the application on
        """
        self.server = multiprocessing.Process(target = app_start,args=(port,))
        self.server.start()
        
    def end_serve(self):
        """
        end the served application
        """
        if self.server is None:
            raise NameError("No server running")
        self.server.terminate()
        self.server.join()
        self.server = None
