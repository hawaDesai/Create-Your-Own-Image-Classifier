# AWS AI/ML Project: Create-Your-Own-Image-Classifier

## Project Title: Image Classifier

### Project Description:
In this project, I focused on:

- Training an image classifier on a dataset of 102 flower categories

- Loading and Preprocessing Data:
  - Imported the necessary packages (PyTorch, NumPy, Matplotlib.pyplot,PIL and torchvisiion) into Jupyter for use throughout the notebook
  - Used torchvision to load the dataset and transform the test, validation and training datasets
    - Transforms were made according to the neeeds of the pre-trained network:
      -  Images Resized to 224x224
      -  Color channels were normalized with the means [0.485, 0.456, 0.406] and for the standard deviations [0.229, 0.224, 0.225]
      -  Random flips and rotation were applied on the validation set
  - JSON was imported to process the mapping of the category names and lEvels
 
  
- Building and training:
  - Used VGG, a torchvission model, as it was the simplest to use for an image classification task
  - New untrained network was built using ReLU activations and dropout
  - The classifier layers were trained using backpropagation based off VGG16
  - Loss and accuracy was tracked to check that the model was being trained and not falling prey to over-training or under-training

- Classifier ran for 10 epochs, small enough to test and run on limited resources

- Model was saved, and rebuilt later on in the Jupyter Notebook

- The model was then used to predict the image it was fed using inference, after being pre-processed using PIL and Pytorch
