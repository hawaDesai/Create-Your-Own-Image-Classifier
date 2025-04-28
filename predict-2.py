import argparse # parser import

%matplotlib inline
%config InlineBackend.figure_format = 'retina'

#numpy and matpotlib imports
import numpy as np
import matplotlib.pyplot as plt

#Torch imports
import torch
from torch import optim
import torch.nn as nn
import torch.nn.functional as F

#Torchvision imports
from torchvision import datasets,transforms, models
from torchvision.datasets import ImageFolder
from torchvision.models import vgg16
import torchvision as torchvision

#PIL imports
from PIL import Image

from train.py import dataload_creat()

'''

functions created:

- parsep_py() : parser function

- load_checkpoint(): loads checkpoint

- process_image(image): processes images in PIL

- imshow(image, ax=None, title=None): shows the image

- predict(image_path, model, topk=5): makes predictions from image

- output(): outputs graph and 5 topk


'''


def parsep_py()

    parser = argparse.ArgumentParser()
    
    parser.add_argument('--save_dir', type = str, default = 'save_directory')
    parser.add_argument('--arch',default = 'vgg16', help = 'accesses the default CNN model architecture') 
    parser.add_argument('--learning_rate', default = '0.01', help = 'defines the learning rate')# default value threee
    parser.add_argument('--hidden_units', default = '512', help = 'the amount of hidden default units')
    parser.add_argument('--epoch', default = '20', help = 'lists epoch number') 
   
    # Replace None with parser.parse_args() parsed argument collection that 
    # you created with this function 
    return parser.parse_args()


def load_checkpoint():
    # TODO: Write a function that loads a checkpoint and rebuilds the model

    load_check = torch.load('checkpoint.pth')
    loaded_checkpoint = nn.Sequential(
                        (nn.Linear(25088,512)),
                        (nn.Dropout(0.2)),
                        (nn.ReLU()),
                        (nn.Linear(512,102)),
                        (nn.ReLU()),
                        (nn.LogSoftmax(dim =1))
                        )

    loaded_checkpoint.load_state_dict(checkpoint['state_dict']) 
    
def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    std = [0,485,0.456,0,406]
    mean = [0.485,0.456,0.406]
    div = 0
    # TODO: Process a PIL image for use in a PyTorch model
    for i in pre_trained_model.class_to_idx:
        with Image.open(I) as i:
            i.resize(256)
            i.CenterCrop(224),
            np_image = np.array(i)
            for val in np_image:
                for valu in means:
                    subt +=val
                    for value in std:
                        div += subt/value

            np_image.append(div)
            ndarray.tranpose(2,0)
            
def imshow(image, ax=None, title=None):
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots()
    
    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.numpy().transpose((1, 2, 0))
    
    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    
    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)
    
    ax.imshow(image)
    
    return ax

def predict(image_path, model, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    # TODO: Implement the code to predict the class from an image file
    prob = torch.exp(model(image_path)) #creates probability
    top_p, top_q = prob.top_k(topk, dim = 1) # looks for first top_k 
    
    plt.x = image_path
    
    plt.y = top_q
    
    plt.bar(x,y)
    
    plt.xlabel('names of flowers')
    plt.ylabel('probability')
    show = plt.show()
    
    return top_p, top_q, show

def output():
    dataiter = iter(dataloader) # dataiter 
    images, labels = dataiter.next() # images, labels create from 
    img = process_image(images[0])

    show = plt.imshow(img)

    pred = predict(cat_to_name, pre_trained_model, topk)
    
    return show,pred
