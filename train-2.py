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

import json



'''

functions created:

- parset_py() : parser function

- dataload_create(data_dir): loads data and returns dataloader to use

- classifier_create(model): creates classifier

- train(epochs, pre_trained_model): trains model

- def save_check(pre_trained_model): saves checkpoint


'''




pre_trained_model_one = models.vgg16(pretrained = True)#pretrain set to boolean value of True
pre_trained_model_two = models.resnet18(pretrained = True)#pretrain set to boolean value of True

models = {'resnet': resnet18, 'vgg': vgg16}

def parset_py():


    parser = argparse.ArgumentParser()
    
  
    parser.add_argument('--top_k', default = '3')
    parser.add_argument('--category_names ',default = 'vgg16', help = 'accesses the default CNN model architecture') 
    parser.add_argument('--learning_rate', default = 'cat_to_name.json', help = 'accesses the file with all the flower names')
    parser.add_argument('--hidden_units', default = '512', help = 'assigns value to hidden units')
    parser.add_argument('--gpu', help = 'allows us to run on the gpu')
   
    # Replace None with parser.parse_args() parsed argument collection that 
    # you created with this function 
    return parser.parse_args()



def dataload_create(data_dir):
    
    
    data_transforms = transforms.Compose([transforms.RandomResizedCrop(224), # resizes to 224 pixels
                                      transofrms.RandomHorizontalFlip(30), # random horizontal flip
                                      transforms.ToTensor(), # creates a tensor
                                      transforms.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225)) # normalizes colors
                                     ])
    image_datasets = ImageFolder(data_dir, transform = data_transforms)
    
    dataloaders = torch.utils.data.DataLoader(image_datasets, batch_size = 64 ,shuffle = True) #shuffling on the training set
    
    return dataloaders

def classifier_create(model):
    
    pre_trained_model = model

    #Freeze parameters
    for parameter in pre_trained_model.parameters():
    parameter.requires_grad = False

    #edit vgg16 classifier

    #beginning of classifier
    classifier = nn.Sequential(
        (nn.Linear(25088,512)), #first layer 
        (nn.Dropout(0.2)), # Dropout function call
        (nn.ReLU()), # call to ReLU
        (nn.Linear(512,102)), # second layer
        (nn.ReLU()), # call to ReLU
        (nn.LogSoftmax(dim =1)) # LogSoftmax function call from torch.nn
    )

    pre_trained_model.classifier = classifier #assigns pre_trained_model classifier 

    loss_criterion = nn.CrossEntropyLoss() #creates a loss_criterion

    optimizer = optim.Adam(pre_trained_model.classifier.parameters(), lr = 0.01) # create the optimizer
    
    


def train(epochs, pre_trained_model):
    
    for e in range(epochs):
    running_loss = 0 #loss from running through training set
    
    #beginning of for loop
    for images,labels in dataloaders:
        optimizer.zero_grad()
        
        train_forward = pre_trained_model.forward(images) # moves forward through the network to train the model
        
        loss = loss_criterion(train_forward, labels) # loss calculator
        
        loss.backward() # goes backword through the network
        
        optimizer.step() # steps through optimizer 

        running_loss += loss.item() #checks loss
        
        pre_trained_model.train() #train the model
        # end of for loop
        
    # beginning of else statement
    else:#referenced from Udacity's transfer learning unit
        test_loss = 0
        test_accuracy = 0
        print('starting')
        with torch.no_grad():#takes out gradient
            pre_trained_model.eval() #evaluates model
            
            # beginning of for loop
            for images, labels in dataloaders:
                
                train_forward = pre_trained_model.forward(images) # restates trainforward with no gradient
                test_loss += loss_criterion(train_forward, labels) # restates testloss with no gradient
                
                class_prob = torch.exp(train_forward) #calculates probability
                top_p, top_class = class_prob.topk(1, dim=1) # 
                
                equals = top_class == labels.view(*top_class.shape) 
                test_accuracy += torch.mean(equals.type(torch.FloatTensor))
            #end of for loop   
    # end of else statement
                
        pre_trained_model.train() #trains model 
        
        #print out validation values, test loss, training loss, test accuracy, accuracy
        print("VALIDATION TEST:\n",
              "Training loss: {:.3f}..".format(training_loss/len(dataloaders)),
              "Test Loss: {:.3f}.. ".format(test_loss/len(dataloaders)),
              "Accuracy: {:.3f}".format(test_accuracy/len(dataloaders)))
        
 def save_check(pre_trained_model):
    pre_trained_model.class_to_idx = image_datasets['train'].class_to_idx,

    checkpoint = {'input_size': 25088,
              'output_size': 102,
              'state_dict': pre_trained_model.state_dict(),
              'classifer': classifier,
              'optimizer': optimizer.state_dict()}

    #save checkpoint:
    torch.save(optimizer.state_dict(),'checkpoint.pth')
    
  