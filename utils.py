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
    This function plots a heatmap of a confusion matrix.
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
    plt.show(block=False)
    plt.pause(30)
    plt.close()

class AccConfPerBin:
    """ 
    This class computes the accuracy and average confidence per bin. 
    Inputs:
    predicted: predicted (torch.Tensor is expected) results from DL model,
    n_bins: the number of bins to compute the calibration error, 
    apply_softmax: whether to apply the softmax function.
    """
    def __init__(self, predicted, n_bins=10, apply_softmax=True):
        self.predicted = predicted
        self.n_bins = n_bins
        self.apply_softmax = apply_softmax 

        if self.apply_softmax:
            self.predicted_prob = softmax(self.predicted,dim=1).data
        else:
            self.predicted_prob = self.predicted.data
        
        self.accuracy, self.index = torch.max(self.predicted_prob,1)
        self.prob = numpy.linspace(0,1,n_bins+1)
            
    def accuracy_per_bin(self, real_tag):
        """" 
        This function computes the accuracy per bin.
        Inputs:
        real_tag (torch.Tensor is expected): ground truth results, 
        Return:
        acc: accuracy per bin,
        prob: probability across bins,
        samples_per_bin: number of samples per bin.
        """

        selected_label = self.index.long() == real_tag
        acc = numpy.linspace(0,1,self.n_bins+1)
        samples_per_bin = []
        for p in range(len(self.prob)-1):
            # Finding elements with probability in between p and p+1
            min_ = self.prob[p]
            max_ = self.prob[p+1]
            boolean_upper = self.accuracy <= max_

            if p == 0: # Including the first element in bin
                boolean_down = self.accuracy >= min_
            else: # Including the previous bin
                boolean_down = self.accuracy > min_

            index_range = boolean_down & boolean_upper
            label_sel = selected_label[index_range]
            
            if len(label_sel) == 0:
                acc[p] = 0.0
            else:
                acc[p] = label_sel.sum()/float(len(label_sel))

            samples_per_bin.append(len(label_sel))

        samples_per_bin = numpy.array(samples_per_bin)
        acc = acc[0:-1]
        prob = self.prob[0:-1]
        
        return acc, prob, samples_per_bin

    def average_confidence_per_bin(self):
        """" 
        This function computes the confidence per bin.
        Return:
        conf: average confidence per bin,
        prob: probability across bins,
        samples_per_bin: number of samples per bin.
        """
        
        conf = numpy.linspace(0,1,self.n_bins+1)
        samples_per_bin = []

        for p in range(len(self.prob)-1):
            # Finding elements with probability in between p and p+1
            min_ = self.prob[p]
            max_ = self.prob[p+1]
            
            boolean_upper = self.accuracy <= max_

            if p == 0: # Including the first element in bin
                boolean_down = self.accuracy >= min_
            else: # Including the previous bin
                boolean_down = self.accuracy > min_

            index_range = boolean_down & boolean_upper
            prob_sel = self.accuracy[index_range]
            
            if len(prob_sel) == 0:
                conf[p] = 0.0
            else:
                conf[p] = prob_sel.sum()/float(len(prob_sel))

            samples_per_bin.append(len(prob_sel))

        samples_per_bin = numpy.array(samples_per_bin)
        conf = conf[0:-1]
        prob = self.prob[0:-1]

        return conf, prob, samples_per_bin

class CalibErrors:
    """" 
    This class computes the Expected Calibration Error (ECE) and Maximum Calibration Error (MCE).
    Inputs:
    acc_bin: accuracy per bin,
    conf_bin: confidence per bin,
    samples_per_bin: number of samples per bin.
    """

    def __init__(self, acc_bin, conf_bin, samples_per_bin):
        self.acc_bin = acc_bin 
        self.conf_bin = conf_bin
        self.samples_per_bin = samples_per_bin 

    def compute_ECE(self):
        """" 
        This function computes the Expected Calibration Error (ECE).
        """
        assert len(self.acc_bin) == len(self.conf_bin)
        ece = 0.0
        total_samples = float(self.samples_per_bin.sum())

        for samples,acc,conf in zip(self.samples_per_bin,self.acc_bin,self.conf_bin):
            ece += samples/total_samples*numpy.abs(acc-conf)

        return ece

    def compute_MCE(self):
        """" 
        This function computes the Maximum Calibration Error (MCE).
        """
        assert len(self.acc_bin) == len(self.conf_bin)

        ce_list = [numpy.abs(self.acc_bin[i] - self.conf_bin[i]) for i in range(len(self.samples_per_bin))]
        mce = numpy.max(ce_list)

        return mce

def plot_calibration_error(acc, conf, prob, ECE, MCE, saving_path):
    """ 
    This function plots the Calibration error across bins.
    Inputs:
    acc: accuracy per bin,
    conf: average confidence per bin, 
    prob: probability across bins,
    ECE: Expected Calibration Error, 
    MCE: Maximum Calibration Error, 
    saving_path: path to save the image.
    """
    plt.figure(figsize=(8, 6))
    plt.bar(prob, abs(acc - conf), 1/len(prob))
    plt.xlim([0, 1.])
    plt.xlabel('Confidence', fontsize=12)
    plt.ylabel('Calibration Error', fontsize=12)

    textstrECE = '$ECE=%.3f$'%(ECE*100) + ' (%)'
    textstrMCE = '$MCE = %.3f$'%(MCE*100) + ' (%)'
    plt.annotate(textstrECE+'\n'+textstrMCE, xy=(0.55, 0.9), xycoords='axes fraction', fontsize=14,
                horizontalalignment='left', verticalalignment='top', bbox=dict(boxstyle='square', facecolor='lightyellow', alpha=0.5))

    plt.title('Calibration error across bins', fontsize=12)
    plt.savefig(saving_path)
    plt.show(block=False)
    plt.pause(30) 
    plt.close()