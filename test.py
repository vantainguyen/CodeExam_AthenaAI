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
from utils import model, plot_confusion_matrix, AccConfPerBin, CalibErrors, plot_calibration_error


# -----------------------------------------------------Define arguments-----------------------------------------------------------------
parser = argparse.ArgumentParser(description='Performance assessment')
parser.add_argument('data', type=str, metavar='', help='A path to a folder containing data to be tested')
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
    1. Perform inference on test dataset
    2. Create and plot a confusion matrix
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

    # Step 3
    ## Converting predicted results and ground truth to torch.Tensor
    y_prob_t = torch.stack(y_prob)
    y_true_t = torch.stack(y_true)

    ## Computing the confidence, accuracy and samples per bin
    acb = AccConfPerBin(y_prob_t, n_bins=10)
    conf = acb.average_confidence_per_bin()[0]
    prob = acb.average_confidence_per_bin()[1]
    samples_per_bin = acb.average_confidence_per_bin()[2]
    acc = acb.accuracy_per_bin(y_true_t)[0]

    ## Computing the Expected calibration error (ECE) and Maximum calibration error (MCE) 
    ce = CalibErrors(acc,conf,samples_per_bin)
    MCE = ce.compute_MCE()
    ECE = ce.compute_ECE()

    ## Creating results folder to save images
    if not os.path.exists(os.path.join(path_base, 'results')):
        os.mkdir(os.path.join(path_base, 'results'))

    saving_path = os.path.join(path_base, 'results', 'calibration_graph.png')

    ## Plotting the Calibration error across bins and saving the figure in the results folder
    plot_calibration_error(acc, conf, prob, ECE, MCE, saving_path)

    # Step 4: Saving false positives from each class
    ## Defining inverse transformation function of a image to its original format
    inv_normalize = transforms.Normalize(
    mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
    std=[1/0.229, 1/0.224, 1/0.225])

    ## Saving folder preparation
    FP_saving_path = os.path.join(path_base, 'results', 'false_positives')
    if not os.path.exists(FP_saving_path):
        os.mkdir(FP_saving_path)

    model.eval()
    j = 0 ## index of saved image

    with torch.no_grad():
        for data, target in dataloaders:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)  ## get the index of the max output
            ## Store wrongly predicted images
            wrong_idx = (pred != target.view_as(pred)).nonzero()[:, 0]
            wrong_samples = data[wrong_idx]
            wrong_preds = pred[wrong_idx]
            actual = target.view_as(pred)[wrong_idx]

            for i in range(len(wrong_idx)):
                sample = wrong_samples[i]
                wrong_pred = wrong_preds[i]
                actual_p = actual[i]
                sample = inv_normalize(sample) ## inversely transform the image to the original format
                img = TF.to_pil_image(sample)
                j += 1
                img.save(os.path.join(FP_saving_path, 'pred{}_actual{}_{}.png'.format(
                    wrong_pred.item(), actual_p.item(), j)))

    # Step 5: Finding potential patterns in the false positives
    ## Loading false positives
    filelist = glob.glob(os.path.join(FP_saving_path, '*.png')) 
    data = np.array([np.array(Image.open(fname)) for fname in filelist])

    ## Transforming data 
    data_do = data.astype('float32') ## Converting to float
    data_do /= 225.0 ## Normalization
    data_do = data_do.reshape(len(data_do), -1) ## Reshaping data to fit in the KMeans method 

    ## KMeans clustering
    kmeans = KMeans(n_clusters = len(class_names))
    kmeans.fit(data_do)

    ## Including labels and image names in a Pandas Dataframe
    image_cluster = pd.DataFrame(filelist,columns=['image'])
    image_cluster["clusterid"] = kmeans.labels_

    ## Saving images into cluster folders
    ## Saving folder preparation
    if not os.path.exists(os.path.join(path_base, 'results', 'false_positivesHINT')):
        os.mkdir(os.path.join(path_base, 'results', 'false_positivesHINT'))

    FP_HINT = os.path.join(path_base, 'results', 'false_positivesHINT')

    for i in range(len(class_names)):
        if not os.path.exists(os.path.join(FP_HINT, 'cluster_{}'.format(i))):
            os.mkdir(os.path.join(FP_HINT, 'cluster_{}'.format(i)))

        ## Images will be seperated according to cluster they belong
        for j in range(len(image_cluster)):
            if image_cluster['clusterid'][j] == i:
                shutil.copy(os.path.join(FP_saving_path, 
                                        image_cluster['image'][j]), os.path.join(FP_HINT, 'cluster_{}'.format(i)))


if __name__ == '__main__':

    main()