""" Authored by: Neel Kanwal (neel.kanwal0@gmail.com)"""

# This file provides inference code for baseline models (CNN+FC) and DKL models, mentioned in the paper.
# Update paths to model weights for running this script

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=DeprecationWarning)
warnings.simplefilter(action='ignore', category=RuntimeWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

import matplotlib.pyplot as plt
font = {'family': 'serif',
        'weight': 'normal',
        'size': 24}
plt.rc('font', **font)

fig = plt.subplots(figsize=(12, 12))

import gpytorch
from torch.autograd import Variable

import pandas as pd
import numpy as np
from datetime import datetime
import seaborn as sns
import os
import time
import json
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler, random_split
from torchvision import datasets, models
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import MultiStepLR, LinearLR, ReduceLROnPlateau, ExponentialLR
import torch.nn.functional as F

from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, matthews_corrcoef, roc_auc_score

import scipy.stats as stats
import statistics

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from my_functions import get_class_distribution, infer_dkl_v2, infer_cnn_v2, DenseNetFeatureExtractor, DKLModel
from my_functions import extract_features, custom_classifier,  count_flops

# from mmcv.cnn.utils import flops_counter
# from fvcore.nn import FlopCountAnalysis
# from ptflops import get_model_complexity_info

torch.cuda.empty_cache()
cuda_device = 1
os.environ["CUDA_VISIBLE_DEVICES"] = str(cuda_device)

NUM_WORKER = 16  # Number of simultaneous compute tasks == number of physical cores
BATCH_SIZE = 64
dropout = 0.2
torch.manual_seed(1700)

#### Selection parameters to define experiment
#### choose dataset, artifact , and model
architecture = "DKL" # "CNN", "DKL"
dataset = "emc" #  "focuspath",  "emc", "tcgafocus", "suh"
artifact = "fold" # Select Blur or Fold artifact
val = False # runs on the validation set of dataset insted of test.

repitions_for_p = 5 # repitions to calculate mean and average across runns

# location where all experiments and models are present.
model_weights = "path_to/DKLModels/weights/"

# blur_cnn_wts = "/DenseNetConfig/04_27_2022 19:47:45"
# blur_dkl_wts = "/04_24_2022 10:26:30" #@ 128
# # blur_dkl_wts = "/04_20_2022 10:09:11" # @256
# # blur_dkl_wts = "/04_29_2022 09:27:26" #(6,6,6) @ 384
# fold_cnn_wts = "/DenseNetConfig/04_27_2022 12:05:05" # 666
# fold_dkl_wts = "/04_21_2022 03:33:31"

## Path to the datasets
if dataset == "focuspath":
    path_to_dataset = "path_to/FocusPath/"
elif dataset == "tcgafocus":
    path_to_dataset = "path_to/tcgafocus/"
elif dataset == "suh":
    path_to_dataset = "path_to/Processed/"
else:
    if artifact == "blur":
        path_to_dataset = "path_to/artifact_dataset/blur/test/"
        if val:
            print("Validation blur subset from EMC")
            path_to_dataset = "path_to/artifact_dataset/blur/validation/"
    elif artifact == "fold":
        path_to_dataset = "path_to/artifact_dataset/fold_20x/test/"
        if val:
            print("Validation fold subset from EMC")
            path_to_dataset = "path_to/artifact_dataset/fold_20x/validation/"
    else:
        print("Dataset does not exists")

# Transform data
test_compose = transforms.Compose([
    transforms.CenterCrop((224, 224)),
    # transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

# Data Loaders to load data.
t = time.time()
print(f"Loading {dataset} - {artifact} Dataset...........")
test_images = datasets.ImageFolder(root=path_to_dataset, transform=test_compose)
total_patches = len(test_images)
idx2class = {v: k for k, v in test_images.class_to_idx.items()}
num_classes = len(test_images.classes)
test_loader = DataLoader(test_images, batch_size=BATCH_SIZE, shuffle=False, sampler=None, num_workers=NUM_WORKER, pin_memory=True)
classes_list = test_loader.dataset.classes
class_distribution = get_class_distribution(test_images)
print("Class distribution in training: ", class_distribution)
print(f"Length of {artifact} testset {len(test_images)} with {num_classes} classes")
print(f"Total data loading time in minutes: {(time.time() - t)/60:.3f}")


now = datetime.now()
date_time = now.strftime("%m_%d_%Y %H:%M:%S")
print(f"Its {date_time}.\n")

# Loading models based on defined experimental setting.
# Best model for blur was DKL with (10,10,10)
# Best model for fold was DKL with (6,6,6)

if architecture == "DKL":
    if artifact == "blur":
        print(f"\nInitializing DKL for {artifact}...............")
        feature_extractor = DenseNetFeatureExtractor(block_config=(10, 10, 10), num_classes=num_classes)
        num_features = feature_extractor.classifier.in_features
        print("Number of output features for patch is ", num_features)
        model = DKLModel(feature_extractor, num_dim=num_features, grid_size=128)
        likelihood = gpytorch.likelihoods.SoftmaxLikelihood(num_features=model.num_dim, num_classes=num_classes)
        pytorch_total_params = sum(p.numel() for p in model.parameters())
        print("Total model parameters (M): ", pytorch_total_params/1e6)
        best_model_wts = model_weights + "/blur_dkl.dat"
        # print("Loading model weights ")
        model.load_state_dict(torch.load(best_model_wts,  map_location=torch.device('cpu'))['model'])
        # print("Loading likelihood weights ")
        likelihood.load_state_dict(torch.load(best_model_wts, map_location=torch.device('cpu'))['likelihood'])

    else:
        print(f"\nInitializing DKL for {artifact}...............")
        feature_extractor = DenseNetFeatureExtractor(block_config=(6, 6, 6), num_classes=num_classes)
        num_features = feature_extractor.classifier.in_features
        print("Number of output features for patch is ", num_features)
        model = DKLModel(feature_extractor, num_dim=num_features, grid_size=128)
        likelihood = gpytorch.likelihoods.SoftmaxLikelihood(num_features=model.num_dim, num_classes=num_classes)
        pytorch_total_params = sum(p.numel() for p in model.parameters())
        print("Total model parameters (M): ", pytorch_total_params/1e6)
        best_model_wts = model_weights + "/fold_dkl.dat"
        # print("Loading model weights ")
        model.load_state_dict(torch.load(best_model_wts, map_location=torch.device('cpu'))['model'])
        # print("Loading likelihood weights ")
        likelihood.load_state_dict(torch.load(best_model_wts, map_location=torch.device('cpu'))['likelihood'])

    input_size = (1, 3, 224, 224)
    flops = count_flops(model, input_size)
    print("GFLOPs:", flops/1e9)


    if torch.cuda.is_available():
        print("Cuda is available")
        model = model.cuda()
        likelihood = likelihood.cuda()

    path = os.path.join('path_to/emc/', f"{dataset}")
    if not os.path.exists(path):
        os.mkdir(path)

    print("\nTesting Starts....................")
    y_pred, y_true, probs, mean1, epistemic, pred_var, lower_1c, upper_1c, feature, entropy = infer_dkl_v2(test_loader, 
        model, likelihood,total_patches=total_patches, n_samples=100)

    file_names = [im[0].split("/")[-1] for im in test_loader.dataset.imgs]
    data = {"files": file_names, "ground_truth": y_true, "prediction": y_pred, "probabilities": probs, \
             "mean1": mean1,"epistemic":epistemic, "variance":pred_var, "lower_conf": lower_1c, "upper_conf": upper_1c, "entropy": entropy}
    dframe = pd.DataFrame(data)

    with pd.ExcelWriter(f"{path}/dkl_predictions_on_{dataset}_for_{artifact}.xlsx") as wr:
        dframe.to_excel(wr, index=False)

    accuracy = accuracy_score(y_true, y_pred)
    print("Accuracy: ", accuracy)
    f1 = f1_score(y_true, y_pred)
    print("F1 Score: ", f1)
    roc = roc_auc_score(y_true, y_pred)
    print("ROC AUC Score: ", roc)
    mathew_corr = matthews_corrcoef(y_true, y_pred)
    print("Mathew Correlation Coefficient: ", mathew_corr)

    acc_list, f1_list, roc_list, mcc_list, pred_list = [],[],[],[],[]
   
    for i in range(repitions_for_p):
        y_pred, y_true, _, _ , _, _, _, _, _, _ = infer_dkl_v2(test_loader, 
        model, likelihood,total_patches=total_patches, n_samples=100)
        accuracy = accuracy_score(y_true, y_pred)
        acc_list.append(accuracy)
        f1 = f1_score(y_true, y_pred)
        f1_list.append(f1)
        roc = roc_auc_score(y_true, y_pred)
        roc_list.append(roc)
        mathew_corr = matthews_corrcoef(y_true, y_pred)
        mcc_list.append(mathew_corr)
        pred_list.append(y_pred)

    print(acc_list)    
    # p_value = stats.ttest_rel(pred_list[0], pred_list[1]).pvalue 
    # print("\nP-value for DKL is", p_value)

    print("\nAccuracy mean: ",statistics.mean(acc_list)," Accuracy std: ", statistics.stdev(acc_list))
    print("\nF1 mean: ",statistics.mean(f1_list)," F1 std: ",statistics.stdev(f1_list))
    print("\nROC mean: ",statistics.mean(roc_list)," ROC std: ",statistics.stdev(roc_list))
    print("\nMCC mean: ",statistics.mean(mcc_list)," MCC std: ",statistics.stdev(mcc_list))

else:
    if artifact == "blur":
        print(f"\nInitializing CNN baseline of DenseNet (10,10,10) for {artifact}...............")
        model = models.DenseNet(block_config = (10,10,10), growth_rate=12, num_init_features=24)
        num_features = model.classifier.in_features# 2208 --> less than 256
        model.classifier = custom_classifier(num_features, num_classes, dropout=dropout)
        print("Number of out features for patch is ", num_features)
        pytorch_total_params = sum(p.numel() for p in model.parameters())
        print("Total model parameters (M): ", pytorch_total_params/1e6)
        # best_model_wts = base_location + blur_cnn_wts
        model.load_state_dict(torch.load(model_weights + "/blur_cnn.dat",map_location=torch.device('cpu'))['model'])
    else:
        print(f"Initializing CNN DenseNet baseline of (6,6,6) Model for {artifact}...............")
        model = models.DenseNet(block_config = (6,6,6), growth_rate=12, num_init_features=24)
        num_features = model.classifier.in_features# 2208 --> less than 256
        model.classifier = custom_classifier(num_features, num_classes, dropout=dropout)
        print("Number of out features for patch is ", num_features)
        pytorch_total_params = sum(p.numel() for p in model.parameters())
        print("Total model parameters (M): ", pytorch_total_params/1e6)
        # best_model_wts = base_location + fold_cnn_wts
        model.load_state_dict(torch.load(model_weights + "/fold_cnn.dat",map_location=torch.device('cpu'))['model'])

    input_size = (1, 3, 224, 224)
    flops = count_flops(model, input_size)
    print("GFLOPs:", flops/1e9)


    if torch.cuda.is_available():
        print("Cuda is available")# model should be on uda before selection of optimizer
        model = model.cuda()

    path = os.path.join('path_to/', f"{dataset}")
    if not os.path.exists(path):
        os.mkdir(path)

    print("\nTesting Starts....................")
    y_pred, y_true, probs, mean1, epistemic, pred_var, lower_1c, upper_1c = infer_cnn_v2(test_loader, model,total_patches=total_patches, n_samples=100)
   
  
    file_names = [im[0].split("/")[-1] for im in test_loader.dataset.imgs]
    data = {"files": file_names, "ground_truth": y_true, "prediction": y_pred, "probabilities": probs, \
             "mean1": mean1, "epistemic": epistemic, "variance":pred_var, "lower_conf": lower_1c, "upper_conf": upper_1c}
    dframe = pd.DataFrame(data)

    with pd.ExcelWriter(f"{path}/cnn_predictions_on_{dataset}_for_{artifact}.xlsx") as wr:
        dframe.to_excel(wr, index=False)


    accuracy = accuracy_score(y_true, y_pred)
    print("Accuracy: ", accuracy)
    f1 = f1_score(y_true, y_pred)
    print("F1 Score: ", f1)
    roc = roc_auc_score(y_true, y_pred)
    print("ROC AUC Score: ", roc)
    mathew_corr = matthews_corrcoef(y_true, y_pred)
    print("Mathew Correlation Coefficient: ", mathew_corr)

    acc_list, f1_list, roc_list, mcc_list, pred_list = [],[],[],[],[]
   
    for i in range(repitions_for_p):
        y_pred, y_true, _, _ , _, _, _, _ = infer_cnn_v2(test_loader, model, total_patches=total_patches, n_samples=100)
        accuracy = accuracy_score(y_true, y_pred)
        acc_list.append(accuracy)
        f1 = f1_score(y_true, y_pred)
        f1_list.append(f1)
        roc = roc_auc_score(y_true, y_pred)
        roc_list.append(roc)
        mathew_corr = matthews_corrcoef(y_true, y_pred)
        mcc_list.append(mathew_corr)
        pred_list.append(y_pred)

    print(acc_list)    
    # p_value = stats.ttest_rel(pred_list[0], pred_list[1]).pvalue 
    # print("\nP-value for CNN is", p_value)

    print("\nAccuracy mean: ",statistics.mean(acc_list)," Accuracy std: ", statistics.stdev(acc_list))
    print("\nF1 mean: ",statistics.mean(f1_list)," F1 std: ",statistics.stdev(f1_list))
    print("\nROC mean: ",statistics.mean(roc_list)," ROC std: ",statistics.stdev(roc_list))
    print("\nMCC mean: ",statistics.mean(mcc_list)," MCC std: ",statistics.stdev(mcc_list))

print("#####################################################")
print(f"Total time in minutes: {(time.time() - t)/60:.3f}")