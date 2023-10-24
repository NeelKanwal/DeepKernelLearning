""" Authored by: Neel Kanwal (neel.kanwal0@gmail.com)"""

# This file provides training code for DKL models proposed in the paper.
# Update paths to processed datasets


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

import seaborn as sns
sns.set_style("white")

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

import os
import time
import math
from datetime import datetime
import json
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets
import torchvision.transforms as transforms
from torch.optim import SGD, Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn.functional as F
from torch import nn
import gpytorch
from my_functions import DenseNetFeatureExtractor, DKLModel, get_class_distribution, epoch_test
from my_functions import train, validate, infer_dkl_v2, count_flops, make_pretty_cm
import pprint
import statistics


from sklearn.metrics import confusion_matrix, accuracy_score, f1_score,  matthews_corrcoef, roc_auc_score
from scikitplot.metrics import plot_roc, plot_precision_recall, plot_lift_curve, plot_ks_statistic, plot_calibration_curve
from scikitplot.helpers import binary_ks_curve
import copy
import random


# Select GPU to run
# cuda_device = 7
# os.environ["CUDA_VISIBLE_DEVICES"] = str(cuda_device)
# torch.cuda.empty_cache()
# device = torch.cuda.current_device()
# print("Current CUDA device, ", device)

BATCH_SIZE = 32
n_epochs = 200
patience = 10
learning_rate = [0.01] #[0.1, 0.01, 0.001]
NUM_WORKER = 32     # Number of simultaneous compute tasks == number of physical cores
opt = [ "Adam"] #["Adam", "SGD"]
lr_scheduler = ["ReduceLROnPlateau"] 
BLOCK_CONFIG = (6, 6, 6) # number of denseblocks with layers inside (6, 6, 6)  (10, 10, 10)  (12, 12, 12)   (16, 16, 16) #  config = (6,12,36,24) for DenseNet161
elbo_beta = 0.5
inducingpoints = 128
test = True
artifact = "blur" # "fold" , "blur"
loop = 5
repeat_for_variance = 5

seeds = []
for i in range(loop):
    seeds.append(random.randint(1, 10000))
print("Random seeds,", seeds)

# Initialize dictionary to collect stats
metrics_val = {'accuracy':[],
           'f1': [],
           'auc':[],
           'mcc':[]}

metrics_test = {'accuracy':[],
           'f1': [],
           'auc':[],
           'mcc':[]}

train_compose = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomRotation(45),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    # transforms.Normalize(mean=(0,),std=(1,))
])
test_compose = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean = [0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

if artifact == "blur":
    path_to_dataset = "path_to/artifact_dataset/blur"# "/home/neel/artifact_dataset/blur" #
elif artifact == "fold":
    path_to_dataset = "path_to/artifact_dataset/fold"#"/home/neel/artifact_dataset/fold" #
else:
    print("Artifact dataset not available")
    raise AssertionError

t = time.time()
print(f"\nLoading {str(artifact)} Dataset...................")
train_images = datasets.ImageFolder(root=path_to_dataset+"/training", transform=train_compose)
idx2class = {v: k for k, v in train_images.class_to_idx.items()}
classes_list = list(idx2class.values())
print("ID to classes ", idx2class)
classes = train_images.classes
class_distribution = get_class_distribution(train_images)
print("Class distribution in training: ", class_distribution)
#Get the class weights. Class weights are the reciprocal of the number of items per class, to obtain corresponding weight for each target sample.
target_list = torch.tensor(train_images.targets)
class_count = [i for i in class_distribution.values()]
print("Class count in training ", class_count)

class_weights = 1./torch.tensor(class_count, dtype=torch.float)
class_weights_all = class_weights[target_list]
train_sampler = WeightedRandomSampler(weights=class_weights_all, num_samples=len(class_weights_all), replacement=True)
train_loader = DataLoader(train_images, batch_size=BATCH_SIZE, sampler=train_sampler, num_workers=NUM_WORKER, pin_memory=True)
print(f"Length of training {len(train_images)} with {len(classes_list)} classes")

val_images = datasets.ImageFolder(root=path_to_dataset+"/validation", transform=test_compose)
total_patches_val = len(val_images)
idx2class = {v: k for k, v in val_images.class_to_idx.items()}
num_classes = len(val_images.classes)
val_loader = DataLoader(val_images, batch_size=BATCH_SIZE, shuffle=False, sampler=None, num_workers=NUM_WORKER, pin_memory=True)
print(f"Length of validation {len(val_images)} with {num_classes} classes")

if test:
    test_images = datasets.ImageFolder(root=path_to_dataset+"/test", transform=test_compose)
    total_patches_test = len(test_images)
    idx2class = {v: k for k, v in test_images.class_to_idx.items()}
    num_classes_ts = len(test_images.classes)
    test_loader = DataLoader(test_images, batch_size=BATCH_SIZE, shuffle=False, sampler=None, num_workers=NUM_WORKER, pin_memory=True)
    print(f"Length of test {len(test_images)} with {num_classes_ts} classes")
print("Total data loading time in minutes: ", (time.time() - t)/60)


## Mulitple training
for seed in range(loop):
    np.random.seed(seeds[seed])
    torch.manual_seed(seeds[seed])
    for sch in lr_scheduler:
        for op in opt:
            for lr in learning_rate:
                print("\n################################################################")
                print(f"Optimizer: {op}   Scheduler: {sch}  Learning rate: {lr}")
                print("##################################################################")
                loss_tr, loss_val, acc_tr, acc_val = [], [], [], []
                t = time.time()
                print(f"RUN: {seed} -Initializing DKL Model...............")
                feature_extractor = DenseNetFeatureExtractor(block_config=BLOCK_CONFIG, num_classes=num_classes)
                num_features = feature_extractor.classifier.in_features
                print("Number of output features for patch is ", num_features)
                model = DKLModel(feature_extractor, num_dim=num_features, grid_size=inducingpoints)
                likelihood = gpytorch.likelihoods.SoftmaxLikelihood(num_features=model.num_dim, num_classes=num_classes)
                pytorch_total_params = sum(p.numel() for p in model.parameters())
                print("Total model parameters (million): ", pytorch_total_params/1e6)
                input_size = (1, 3, 224, 224)
                flops = count_flops(model, input_size)
                print("GFLOPs:", flops/1e9)

                if torch.cuda.is_available():
                    print("Cuda is available")
                    model = model.cuda()
                    likelihood = likelihood.cuda()
                # One way to penalize complexity, would be to add all our parameters (weights) to our loss function.
                # weights deacay is some of square of all weights added to the loss
                #  if you have too much weight decay, then no matter how much you train,
                #  the model never quite fits well enough whereas if you have too little weight decay,
                #  you can still train well, you just have to stop a little bit early.
                # If you need to move a model to GPU via .cuda(), please do so before constructing optimizers for i
                mll = gpytorch.mlls.VariationalELBO(likelihood, model.gp_layer, num_data=len(train_loader.dataset), beta=elbo_beta)

                if op == "SGD":
                    optimizer = SGD([
                        {'params': model.feature_extractor.parameters(), 'weight_decay': 1e-4},
                        {'params': model.gp_layer.hyperparameters(), 'lr': lr * 0.01},
                        {'params': model.gp_layer.variational_parameters()},
                        {'params': likelihood.parameters()}], lr=lr, momentum=0.9, nesterov=True, weight_decay=0)

                elif op == "Adam":
                    optimizer = Adam([
                        {'params': model.feature_extractor.parameters(), 'weight_decay': 1e-2},
                        {'params': model.gp_layer.hyperparameters(), 'lr': lr * 0.01},
                        {'params': model.gp_layer.variational_parameters()},
                        {'params': likelihood.parameters()}], lr=lr, betas=(0., 0.95), eps=1e-8, weight_decay=0)

                else:
                    print("Optimizer does not exists in settings.\n")

                if sch == "ReduceLROnPlateau":
                    # Reduce learning rate when a metric has stopped improving.
                    scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=5, verbose=True)
                else:
                    print("Scheduler does not exists in settings.\n")

                print("\nTraining Starts....................\n")
                now = datetime.now()
                date_time = now.strftime("%m_%d_%Y %H:%M:%S")
                print(f"\nFiles for this experiment will be saved with {date_time} time stamp directory")

                if not os.path.exists(os.path.join(os.getcwd(), "experiments", "DKL", date_time)):
                    if not os.path.exists(os.path.join(os.getcwd(), "experiments", "DKL")):
                        os.mkdir(os.path.join(os.getcwd(), "experiments", "DKL"))
                    path = os.path.join(os.getcwd(), "experiments", "DKL", date_time)
                    os.mkdir(path)
                    print(f"\nDirectory Created {path}.")

                param_dict = {
                              "BATCH_SIZE": BATCH_SIZE,
                              "EPOCHS": n_epochs,
                              "PATIENCE": patience,
                              "Learning Rate": learning_rate,
                              "NETWORK CONFIG": BLOCK_CONFIG,
                              "Optimizer": op,
                              "LR Scheduler": sch,
                              "MLL": elbo_beta,
                              "Artifact": artifact}

                pprint.pprint(param_dict)
                with open(f"{path}/Parameters.json", "a+") as f:
                    json.dump(param_dict, f, indent=4)

                tr_acc, tr_loss = epoch_test(model,likelihood, mll, train_loader)
                val_acc, val_loss = epoch_test(model, likelihood, mll, val_loader)
                print("\nEpoch 0")
                print("\nValidation accuracy : {0:.3f} %\n".format(val_acc))
                loss_val.append(val_loss)
                loss_tr.append(tr_loss)
                acc_val.append(val_acc)
                acc_tr.append(tr_acc)


                # training loop
                epoch_finished = 0
                best_model_wts = copy.deepcopy(model.state_dict())
                best_likelihood_wts = copy.deepcopy(likelihood.state_dict())
                best_acc = 0.0
                counter = 0
                for epoch in range(1, n_epochs + 1):
                    with gpytorch.settings.use_toeplitz(False):
                        # with gpytorch.settings.use_toeplitz(False), gpytorch.settings.max_preconditioner_size(0):
                        tr_acc, tr_loss = train(epoch, train_loader, model, likelihood, optimizer, mll)
                        val_acc, val_loss = validate(epoch, val_loader, model, likelihood, mll)
                        loss_val.append(val_loss)
                        loss_tr.append(tr_loss)
                        acc_val.append(val_acc)
                        acc_tr.append(tr_acc)
                        epoch_finished += 1
                        counter += 1     
                        if val_acc > best_acc:
                            best_acc = val_acc
                            best_model_wts = copy.deepcopy(model.state_dict())
                            best_likelihood_wts = copy.deepcopy(likelihood.state_dict())
                        if counter >=patience:
                            print(f"Early stopping at epoch {epoch}...\n")
                            break
                    if sch == "ReduceLROnPlateau":
                        scheduler.step(val_loss)
                    else:
                        scheduler.step()
                print(f"Run:{i} -- Total training time for {epoch_finished} epochs in minutes: ", (time.time() - t)/60)
                print(f"Run:{i} -- Best accuracy for {str(artifact)} is {best_acc:.3f} %.")
                #
                torch.save({'model': best_model_wts, 'likelihood': best_likelihood_wts}, f"{path}/best_weights.dat")
                plt.clf()
                plt.figure(1)
                plt.plot(loss_tr, "goldenrod", label="Training loss")
                plt.plot(loss_val, "slategray", label="Validation loss")
                plt.title(f"Loss Curve for {str(artifact)} classification.")
                plt.legend(loc="best")
                plt.savefig(f"{path}/Loss Curve for {str(artifact)}.png")

                plt.clf()
                plt.figure(2)
                plt.plot(acc_tr, "indianred", label="Training accuracy")
                plt.plot(acc_val, "goldenrod", label="Validation accuracy")
                plt.title(f"Accuracy Curve for {str(artifact)} classification.")
                plt.legend(loc="best")
                plt.savefig(f"{path}/Accuracy for {str(artifact)}.png")

                with open(f"{path}/Experimental Values.txt", "a+", encoding='utf-8') as f:
                    acc_list_tr = [a.tolist() for a in acc_tr]
                    acc_list_val = [a.tolist() for a in acc_val]
                    dict = {"training_loss": loss_tr, "validation_loss": loss_val, "training_accuracy": acc_list_tr, "validation_accuracy": acc_list_val}
                    f.write(str(dict))

                print(f"\nBest model weights with accuracy {best_acc:.3f} % loaded to compute metrices.....\n")
                model.load_state_dict(best_model_wts)
                likelihood.load_state_dict(best_likelihood_wts)

                print("----------------------------------")
                print("\n########## Validation ##########")
                y_pred, y_true, probs, mean1, epistemic, pred_var, lower_1c, upper_1c, feature, entropy = infer_dkl_v2(val_loader, 
                    model, likelihood,total_patches=total_patches_val, n_samples=100)

                file_names = [im[0].split("/")[-1] for im in val_loader.dataset.imgs]
                data = {"files": file_names, "ground_truth": y_true, "prediction": y_pred, "probabilities": probs, \
                         "mean1": mean1,"epistemic":epistemic, "variance":pred_var, "lower_conf": lower_1c, "upper_conf": upper_1c, "entropy": entropy}
                dframe = pd.DataFrame(data)

                with pd.ExcelWriter(f"{path}/dkl_predictions_val.xlsx") as wr:
                    dframe.to_excel(wr, index=False)

                accuracy = accuracy_score(y_true, y_pred)
                print("Accuracy: ", accuracy)
                f1 = f1_score(y_true, y_pred)
                print("F1 Score: ", f1)
                roc = roc_auc_score(y_true, y_pred)
                print("ROC AUC Score: ", roc)
                mathew_corr = matthews_corrcoef(y_true, y_pred)
                print("Mathew Correlation Coefficient: ", mathew_corr)

                metrics_val['accuracy'].append(accuracy)
                metrics_val['f1'].append(f1)
                metrics_val['auc'].append(roc)
                metrics_val['mcc'].append(mathew_corr)


                acc_list, f1_list, roc_list, mcc_list,pred_list = [],[],[],[],[]
                for i in range(repeat_for_variance):
                    y_pred, y_true, _, _ , _, _, _, _, _, _ = infer_dkl_v2(val_loader, 
                    model, likelihood,total_patches=total_patches_val, n_samples=100)
                    accuracy = accuracy_score(y_true, y_pred)
                    acc_list.append(accuracy)
                    f1 = f1_score(y_true, y_pred)
                    f1_list.append(f1)
                    roc = roc_auc_score(y_true, y_pred)
                    roc_list.append(roc)
                    mathew_corr = matthews_corrcoef(y_true, y_pred)
                    mcc_list.append(mathew_corr)
                    pred_list.append(y_pred)

                cm = confusion_matrix(y_true, y_pred)
                print(cm)

                make_pretty_cm(cm, categories=classes_list, title=f"DKL_validation_{artifact}")
                plt.savefig(f"{path}/DKL_validation_{artifact}.png")    

                print(f"\n----run {seed} Results from 5 runs of Validation----")        
                print("\nAccuracy mean: ",statistics.mean(acc_list)," Accuracy std: ", statistics.stdev(acc_list))
                print("\nF1 mean: ",statistics.mean(f1_list)," F1 std: ",statistics.stdev(f1_list))
                print("\nROC mean: ",statistics.mean(roc_list)," ROC std: ",statistics.stdev(roc_list))
                print("\nMCC mean: ",statistics.mean(mcc_list)," MCC std: ",statistics.stdev(mcc_list))

                # Repeating it for test set.

                print("----------------------------------")
                print("\n################ Test ##########")
                y_pred, y_true, probs, mean1, epistemic, pred_var, lower_1c, upper_1c, feature, entropy = infer_dkl_v2(test_loader, 
                    model, likelihood,total_patches=total_patches_test, n_samples=100)

                file_names = [im[0].split("/")[-1] for im in test_loader.dataset.imgs]
                data = {"files": file_names, "ground_truth": y_true, "prediction": y_pred, "probabilities": probs, \
                         "mean1": mean1,"epistemic":epistemic, "variance":pred_var, "lower_conf": lower_1c, "upper_conf": upper_1c, "entropy": entropy}
                dframe = pd.DataFrame(data)

                with pd.ExcelWriter(f"{path}/dkl_predictions_test.xlsx") as wr:
                    dframe.to_excel(wr, index=False)

                accuracy = accuracy_score(y_true, y_pred)
                print("Accuracy: ", accuracy)
                f1 = f1_score(y_true, y_pred)
                print("F1 Score: ", f1)
                roc = roc_auc_score(y_true, y_pred)
                print("ROC AUC Score: ", roc)
                mathew_corr = matthews_corrcoef(y_true, y_pred)
                print("Mathew Correlation Coefficient: ", mathew_corr)

                metrics_test['accuracy'].append(accuracy)
                metrics_test['f1'].append(f1)
                metrics_test['auc'].append(roc)
                metrics_test['mcc'].append(mathew_corr)


                acc_list, f1_list, roc_list, mcc_list,pred_list = [],[],[],[],[]
                for i in range(repeat_for_variance):
                    y_pred, y_true, _, _ , _, _, _, _, _, _ = infer_dkl_v2(test_loader, 
                    model, likelihood,total_patches=total_patches_test, n_samples=100)
                    accuracy = accuracy_score(y_true, y_pred)
                    acc_list.append(accuracy)
                    f1 = f1_score(y_true, y_pred)
                    f1_list.append(f1)
                    roc = roc_auc_score(y_true, y_pred)
                    roc_list.append(roc)
                    mathew_corr = matthews_corrcoef(y_true, y_pred)
                    mcc_list.append(mathew_corr)
                    pred_list.append(y_pred)

                cm = confusion_matrix(y_true, y_pred)
                print(cm)
                make_pretty_cm(cm, categories=classes_list, title=f"DKL_test_{artifact}")
                plt.savefig(f"{path}/DKL_test_{artifact}.png")    

                #multiple runs to find mean and std across metrics.

                print(f"\n----run {seed} Results from 5 runs of Test----")        
                print("\nAccuracy mean: ",statistics.mean(acc_list)," Accuracy std: ", statistics.stdev(acc_list))
                print("\nF1 mean: ",statistics.mean(f1_list)," F1 std: ",statistics.stdev(f1_list))
                print("\nROC mean: ",statistics.mean(roc_list)," ROC std: ",statistics.stdev(roc_list))
                print("\nMCC mean: ",statistics.mean(mcc_list)," MCC std: ",statistics.stdev(mcc_list))


    with open(f"{path}/matrics_val_{artifact}_dkl.json", "a+") as f:
        json.dump(metrics_val, f, indent=4)

    with open(f"{path}/matrics_test_{artifact}_dkl.json", "a+") as f:
        json.dump(metrics_test, f, indent=4)

# Show values from five runs to calculate ASO test using aso_test.py
print("\nMetrics for all runs (val): ")
print(metrics_val)
print("\nMetrics for all runs (test): ")
print(metrics_test)

print("\n--------------------------------------------")
print(f"Program finished for DKL {BLOCK_CONFIG}.......")
print("--------------------------------------------")
