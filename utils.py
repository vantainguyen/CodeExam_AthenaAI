import torch
import os
import torchvision
import torch.nn as nn 
from torch.nn.functional import softmax
import seaborn as sns
import matplotlib.pyplot as plt 
import pandas as pd
import numpy 


def model(model_type, path_weights):

    """  
    Loading the given ConvNext architecture and its pre-trained weights for Cifar10 dataset.
    Inputs:
    model_type: the type of ConveNext model 'tiny', 'small', 'large' and 'base',
    path_weights: path to pre-trained weights of ConvNext for Cifar10 dataset.
    """
    # Loading a pre-trained ConvNext model
    if model_type == 'tiny':
        model_conv = torchvision.models.convnext_tiny(pretrained=False)
    elif model_type == 'small':
        model_conv = torchvision.models.convnext_small(pretrained=False)
    elif model_type == 'base':
        model_conv = torchvision.models.convnext_base(pretrained=False)
    elif model_type == 'large':
        model_conv = torchvision.models.convnext_large(pretrained=False)

    # Modifying the classifier layer
    num_ftrs = model_conv.classifier[-1].in_features
    model_conv.classifier[-1] = nn.Sequential(nn.Linear(num_ftrs, 10)) 

    # Assigning model to a GPU or a CPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model_conv = model_conv.to(device)

    weight_name = os.path.join(path_weights, 'ConvNext' + model_type.capitalize() + 'Cifar10.hdf5')
    model_conv.load_state_dict(torch.load(weight_name, map_location=torch.device(device.type))) 

    return model_conv

def plot_confusion_matrix(cf_matrix, class_names):

    """
    This function plot a heatmap of a confusion matrix.
    Inputs:
    cf_matrix: Confusion matrix, 
    class_names: names of classes.
    """
    # Create pandas dataframe
    dataframe = pd.DataFrame(cf_matrix, index=class_names, columns=class_names)
    
    # Create confusion matrix heatmap
    plt.figure(figsize=(8, 6))
    hm = sns.heatmap(dataframe, annot=True, cbar=True,cmap="YlGnBu",fmt="d")
    hm.set_xticklabels(hm.get_xticklabels(), rotation=40)
    plt.title("Confusion Matrix", fontsize=12)
    plt.ylabel("True Class", fontsize=12)
    plt.xlabel("Predicted Class", fontsize=12)
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    plt.show(block=False)
    plt.pause(30)
    plt.close()