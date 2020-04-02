import numpy as np
import torch
import torch.nn as nn


class Projector():
    def __init__(self,layers,N_in):
        self.N_in = N_in
        self.layers = layers
        self.projs = self.get_chained_proj()
        
    @staticmethod
    def N_out(K,P,S,N_in):
        """
        return the output dimension given the input one for Conv2d or MaxPool2d
        """
        return (int((N_in+2*P-K)/S)+1)
    
    @staticmethod
    def get_slic (slic,S,P,K,N):
        """
        returns the equivalent slice for the input tensor of the filter
        Args:
            slic(tuple(tuple(int))): slices for the output of the filter
            S(int): stride
            P(int): padding
            K(int): kernel
            N(int): side size of the input tensor
        """
        assert(isinstance(slic[0],int) and isinstance(slic[1],int))
        assert(isinstance(S,int) and isinstance(K,int) and isinstance(P,int))
        return (max(slic[0]*S-P,0),min(slic[1]*S +K-P,N))
    
    def get_deprojecter(self,layer,n):
        """
        return the projector for the given layer
        Args:
            layer(nn.Conv2d or nn.MaxPool2d): the layer you want a projecter for
            n(int): the side size of the input tensor (assumed to be squared) i.e the tensor
            is assumed to be (1x3xnxn)
        """
        K = layer.kernel_size
        P = layer.padding
        S = layer.stride
        if (isinstance(layer,nn.MaxPool2d)):
            return (lambda slices:(
                        Projector.get_slic(slices[0],S,P,K,n),
                        Projector.get_slic(slices[1],S,P,K,n)
                    ),
                    Projector.N_out(K,P,S,n)) #TODO: dont assume square image
        else:
            return (lambda slices:(
                        Projector.get_slic(slices[0],S[0],P[0],K[0],n),
                        Projector.get_slic(slices[1],S[1],P[1],K[1],n)
                    ),
                    Projector.N_out(K[0],P[0],S[0],n)) #TODO: dont assume square image
    
    def get_chained_proj(self):
        """
        Returns the projectors that will be used for the chaining
        """
        #non recursive call
        N = self.N_in
        projs = []
        for layer in self.layers:
            proj,N_out = self.get_deprojecter(layer,N)
            projs.append(proj)
            N = N_out
        return projs
    
    def chain(self,slices):
        """
        return the slice in the original image which induced the slice slices
        Args:
            slices(tuple(tuple(int))): the slices in the output as ((x1,x2),(y1,y2))
        """
        for proj in self.projs[::-1]:
            slices = proj(slices)
        return slices
    
    def viz(self,slices):
        """
        Visualize the slices effects amongst all the intermediate filters for the given Projector
        Args:
            slices(tuple(tuple(int))): the slices in the output as ((x1,x2),(y1,y2))
        """
        #layers_copy = deepcopy(self.layers)
        self.layers_copy = self.layers
        imgs = [torch.zeros([1,3,self.N_in,self.N_in])]
        
        for layer in self.layers:
            if isinstance(layer,nn.Conv2d):
                layer2 = nn.Conv2d(3,3,layer.kernel_size,layer.stride,layer.padding)
                imgs.append(layer2(imgs[-1]))
            else:
                imgs.append(layer(imgs[-1]))
        
        assert(len(self.projs) == len(imgs)-1)
        for proj,img in zip(self.projs[::-1],imgs[::-1]):
            (x1,x2),(y1,y2) = slices
            img[0,:,x1:x2+1,y1:y2+1] = 255
            slices = proj(slices)
        (x1,x2),(y1,y2) = slices
        imgs[0][0,:,x1:x2+1,y1:y2+1] = 255
        
        dim = int(np.floor(np.sqrt(len(self.layers))))+1
        fig,axes = plt.subplots(dim,dim,figsize=(10,10))
        for i,img in enumerate(imgs):
            a,b = np.unravel_index(i,(dim,dim))
            axes[a,b].imshow((img[0].detach().permute(1,2,0).numpy()).astype(np.uint8))
            axes[a,b].set_title(str(i))