from __future__ import print_function, division
import torch
import torchvision.transforms.functional as TF
import glob
import numpy as np
from torchvision import datasets, transforms
import os
from sklearn.metrics import confusion_matrix 
from sklearn.cluster import KMeans
import pandas as pd
import shutil
from PIL import Image
import argparse
from utils import model, plot_confusion_matrix

# -----------------------------------------------------Define arguments-----------------------------------------------------------------
parser = argparse.ArgumentParser(description='Performance assessment')
parser.add_argument('data', type=str, default='test', metavar='', help='A path to a folder containing data to be tested')
parser.add_argument('--model', type=str, default='tiny', metavar='', help="Type of ConvNext model: 'tiny', 'small', 'large' and 'base'")

args = parser.parse_args()

# ---------------------------------------------------------------------------------------------------------------------------------------

# Defining transformation function
data_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

# Loading and transforming the testing dataset
image_datasets = datasets.ImageFolder(args.data, data_transforms)
                  
dataloaders = torch.utils.data.DataLoader(image_datasets, batch_size=4,
                                             shuffle=True, num_workers=4)
             
class_names = image_datasets.classes # Obtaining names of classes

model_type = args.model.lower()

path_base = os.getcwd()
path_weights = os.path.join(path_base, 'pretrained_weights')

# Initializing a ConvNext model
model = model(model_type, path_weights)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def main():

    """
    This script performs following steps to assess the performance of a classification model based on ConvNext.
    1. Perform inference on testing dataset
    2. Create and plote a confusion matrix
    3. Compute and plot Calibration error
    4. Save false positives from each class
    5. Find potential patterns in the false positives
    """
    # Step 1
    model.eval() ## Set model to evaluation mode

    y_true = []
    y_pred = []
    y_prob = []

    with torch.no_grad():
        for data, target in dataloaders:
            data, target = data.to(device), target.to(device)

            y_true.extend(target.cpu())

            output = model(data)
            y_prob.extend(output.cpu())

            _, predicted = torch.max(output, 1)
            y_pred.extend(predicted.cpu())

    # Step 2
    ## Create a confusion matrix
    cf_matrix = confusion_matrix(y_true, y_pred)
 
    ## Plot a heatmap of the confusion matrix
    plot_confusion_matrix(cf_matrix, class_names)


if __name__ == '__main__':

    main()