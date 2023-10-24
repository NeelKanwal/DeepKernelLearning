""" Authored by: Neel Kanwal (neel.kanwal0@gmail.com)"""

# This file provides helpful functions for python files in this repository.

# You can also use this implementation from torchvision to train your models
# from torchvision.models import DenseNet 

# or Alternative Implemenetaion from DenseNet-BC from https://github.com/pytorch/vision/blob/master/torchvision/models/densenet.py
# Following denseNet file borrowed from :https://docs.gpytorch.ai/en/stable/examples/06_PyTorch_NN_Integration_DKL/Deep_Kernel_Learning_DenseNet_CIFAR_Tutorial.html
from densenet import DenseNet

import gpytorch
import pandas as pd
import sys
import numpy as np
import seaborn as sns
import torch
import math
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from torchvision import datasets
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from torch.autograd import Variable
import scipy.stats as stats
import time

# Reference : Gpytorch documentation
class DenseNetFeatureExtractor(DenseNet):
    """ DenseNet feature extractor to take image and provide a feature vector for GP Classifier
    """
    def forward(self, x):
        features = self.features(x)
        out = F.relu(features, inplace=True)
        #out = F.avg_pool2d(out, kernel_size=self.avgpool_size).view(features.size(0), -1) # only works for inputs of 32 x 32
        out = F.adaptive_avg_pool2d(out, output_size=(1, 1)).view(features.size(0), -1)
        return out


class GaussianProcessLayer(gpytorch.models.ApproximateGP):
    """ using one GP per feature, and mixing them in softmax likelihood.
        the Gaussian process layer
    """
    def __init__(self, num_dim, grid_bounds=(-10., 10.), grid_size=512):
        variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(num_inducing_points=grid_size, batch_shape=torch.Size([num_dim]))
        # We wrap GridInterpolationVariationalStrategy with a MultitaskVariationalStrategy so that our output is a vector-valued GP
        variational_strategy = gpytorch.variational.IndependentMultitaskVariationalStrategy(
            gpytorch.variational.GridInterpolationVariationalStrategy(
                self, grid_size=grid_size, grid_bounds=[grid_bounds],
                variational_distribution=variational_distribution,
            ), num_tasks=num_dim,)
        super().__init__(variational_strategy)

        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(
                lengthscale_prior=gpytorch.priors.SmoothedBoxPrior(
                    math.exp(-1), math.exp(1), sigma=0.1, transform=torch.exp
                )
            )
            )
        self.mean_module = gpytorch.means.ConstantMean()
        self.grid_bounds = grid_bounds

    def forward(self, x):
        mean = self.mean_module(x)
        covar = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean, covar)

# Reference : Gpytorch documentation
# SVDKL Model
class DKLModel(gpytorch.Module):
    """ Combing above feature extracto and GP layer to create a single probablistic model
    """
    def __init__(self, feature_extractor, num_dim, grid_bounds=(-10., 10.), grid_size=512):
        super(DKLModel, self).__init__()
        self.feature_extractor = feature_extractor
        self.gp_layer = GaussianProcessLayer(num_dim=num_dim, grid_bounds=grid_bounds, grid_size=grid_size)
        self.grid_bounds = grid_bounds
        self.num_dim = num_dim
        self.scale_to_bounds = gpytorch.utils.grid.ScaleToBounds(self.grid_bounds[0], self.grid_bounds[1])

    def forward(self, x):
        features = self.feature_extractor(x)
        features = self.scale_to_bounds(features)
        # This next line makes it so that we learn a GP for each feature
        features = features.transpose(-1, -2).unsqueeze(-1)
        res = self.gp_layer(features)
        return res


def get_class_distribution(dataset_obj):
    """ To see number of patches in each class
    Input: Dataset Object
    Output: Dictionary
    """
    count_dict = {k:0 for k,v in dataset_obj.class_to_idx.items()}
    idx2class = {v: k for k, v in dataset_obj.class_to_idx.items()}
    for element in dataset_obj:
        y_lbl = element[1]
        y_lbl = idx2class[y_lbl]
        count_dict[y_lbl] += 1
    return count_dict


def train(epoch, train_loader, model, likelihood, optimizer, mll):
    """ Used for training DKL model in train_dkl.py
    """
    model.train()
    likelihood.train()
    train_losses = []
    correct = 0
    print(f"Training epoch: {epoch}")
    with gpytorch.settings.num_likelihood_samples(16), gpytorch.settings.cholesky_jitter(1e-1):
        for data, target in train_loader:
            if torch.cuda.is_available():
                data, target = data.cuda(), target.cuda()
            optimizer.zero_grad()
            output = model(data)
            loss = -mll(output, target)
            train_losses.append(loss.item())
            # loss_m.append(loss)
            loss.backward()
            optimizer.step()

            output_pred = likelihood(model(data))
            pred = output_pred.probs.mean(0).argmax(-1)  # Taking the mean over all of the sample we've drawn
            correct += pred.eq(target.view_as(pred)).cpu().sum()
        train_accuracy = (100. * correct / len(train_loader.dataset)).cpu().detach().numpy()
        train_loss = np.average(train_losses)
    return train_accuracy, train_loss

def validate(epoch, test_loader, model, likelihood, mll):
     """ Used for validating DKL model in train_dkl.py
    """
    model.eval()
    likelihood.eval()
    valid_losses = []
    correct = 0
    with torch.no_grad(),  gpytorch.settings.cholesky_jitter(1e-1), gpytorch.settings.fast_pred_var():
        # gpytorch.settings.num_likelihood_samples(8),
        # Fast predictive variances using Lanczos Variance Estimates (LOVE) Use this for improved performance when computing predictive variances.
        # The number of samples to draw from a latent GP when computing a likelihood This is used in variational inference and training
        for data, target in test_loader:
            if torch.cuda.is_available():
                data, target = data.cuda(), target.cuda()
            output = model(data)  # This gives us 16 samples from the predictive distribution
            loss = -mll(output, target)
            valid_losses.append(loss.item())
            output_pred = likelihood(model(data))
            pred = output_pred.probs.mean(0).argmax(-1)  # Taking the mean over all of the sample we've drawn
            correct += pred.eq(target.view_as(pred)).cpu().sum()
            # minibatch_iter.set_postfix(loss=loss.item())

    val_accuracy = (100. * correct / float(len(test_loader.dataset))).cpu().detach().numpy()
    valid_loss = np.average(valid_losses)
    print("Validation accuracy {0:.3f} %\n".format(val_accuracy))
    return val_accuracy, valid_loss

def epoch_test(model, likelihood, mll, loader):
    """ Used for running DKL model before training to test Zeroth epoch in train_dkl.py
    """
    model.eval()
    likelihood.eval()
    valid_losses = []
    correct = 0
    with torch.no_grad(),  gpytorch.settings.cholesky_jitter(1e-1), gpytorch.settings.fast_pred_var():
        for data, target in loader:
            if torch.cuda.is_available():
                data, target = data.cuda(), target.cuda()
            output = model(data)
            loss = -mll(output, target)
            valid_losses.append(loss.item())

            output_pred = likelihood(model(data))
            # probabilities = F.softmax(output_pred, dim=1)
            pred = output_pred.probs.mean(0).argmax(-1)  # Taking the mean over all of the sample we've drawn
            correct += pred.eq(target.view_as(pred)).cpu().sum()

    val_accuracy = (100. * correct / float(len(loader.dataset))).detach().cpu().numpy()
    valid_loss = np.average(valid_losses)
    return val_accuracy, valid_loss

def dummy_data(BATCH_SIZE, train_compose, test_compose):
    """ Dummy dataset used for training models with CIFAR10
    """
    print("Loading CIFAR10 as dummy data.")
    train_set = datasets.CIFAR10('data', train=True, transform=train_compose, download=True)
    test_set = datasets.CIFAR10('data', train=False, transform=test_compose)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_set, BATCH_SIZE, shuffle=False)
    return train_loader, test_loader


def convert_batch_list(lst_of_lst):
    """ Make a single list from list of lists
        Used in training, validation and inference functions to create excel sheet
    """
    return sum(lst_of_lst, [])


# rows to be the “true class” and the columns to be the “predicted class.”
def make_cm(targets_list, predictions_list, classes):
    # labels = [‘True Neg’,’False Pos’,’False Neg’,’True Pos’]
    cm = confusion_matrix(targets_list, predictions_list)
    confusion_matrix_df = pd.DataFrame(cm, columns=classes, index=classes)
    fig = plt.figure(figsize=(12, 10))
    fig = sns.heatmap(confusion_matrix_df, annot=True, fmt= "d", cmap= "coolwarm")
    fig.set(ylabel = "True", xlabel="Predicted", title='DKL predictions' )
    plt.xticks(rotation=45)
    plt.yticks(rotation=45)
    return cm




def train_cnn_v2(model, criterion, optimizer, train_loader, epoch):
    # Used in Baseline Models
    model.train()
    train_losses = []
    correct = 0
    print(f"Training epoch: {epoch}")
    for data, target in train_loader:
        if torch.cuda.is_available():
            data, target = data.cuda(), target.cuda()
        try:
            output, _ = model(data)
        except:
            output = model(data)

        optimizer.zero_grad()

        try:
            _, preds = torch.max(output, 1)
        except: 
            _, preds = torch.max(output[0], 1)

        try:
            loss = criterion(output, target)
        except: 
            loss = criterion(output[0], target)

        loss.backward()
        train_losses.append(loss.item())
        optimizer.step()
        correct += preds.eq(target.view_as(preds)).cpu().sum()
    train_accuracy = (100. * correct / float(len(train_loader.dataset))).cpu().detach().numpy()
    train_loss = np.average(train_losses)
    # print("Training accuracy: {0:.3f} %\n".format(train_accuracy))
    return train_accuracy, train_loss


def val_cnn_v2(model, test_loader, criterion, epoch):
    with torch.no_grad():
        model.eval()
        valid_losses = []
        correct = 0
        stop = False
        for data, target in test_loader:
            if torch.cuda.is_available():
                data, target = data.cuda(), target.cuda()
            try:
                output, _ = model(data)
            except:
                output = model(data)

            try:
                _, preds = torch.max(output, 1)
            except: 
                _, preds = torch.max(output[0], 1)

            try:
                loss = criterion(output, target)
            except: 
                loss = criterion(output[0], target)
            
            valid_losses.append(loss.item())
            correct += preds.eq(target.view_as(preds)).cpu().sum()
        val_accuracy = (100. * correct / float(len(test_loader.dataset))).detach().cpu().numpy()
        valid_loss = np.average(valid_losses)
        print("Validation accuracy: {0:.3f} %\n".format(val_accuracy))
        return val_accuracy, valid_loss

def epoch_test_cnn_v2(model, loader, criterion):
        with torch.no_grad():
            model.eval()
            valid_losses = []
            correct = 0
            for data, target in loader:
                if torch.cuda.is_available():
                    data, target = data.cuda(), target.cuda()
                try:
                    output, _ = model(data)
                except:
                    output = model(data)

                _, preds = torch.max(output, 1)
                loss = criterion(output, target)
                # print(loss)
                try:
                    valid_losses.append(loss.item())
                except:
                    valid_losses.append(loss)
                correct += preds.eq(target.view_as(preds)).cpu().sum()

            val_accuracy = (100. * correct / float(len(loader.dataset))).detach().cpu().numpy()
            # print(valid_losses)
            valid_loss = np.average(valid_losses)
            return val_accuracy, valid_loss


class custom_classifier(nn.Module):
    def __init__(self, in_features, num_classes, dropout=0.2):
        super(custom_classifier, self).__init__()
        self.fc1 = nn.Linear(in_features, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        x = F.relu(self.fc1(x)) # fully connected layer 1
        x = self.dropout(x)
        feat = F.relu(self.fc2(x)) # fully connected layer 2
        x = self.dropout(x)
        x = self.fc3(feat)   #fully connected layer 3
        return x, feat

class FocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=0.25, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha, (float, int)): self.alpha = torch.Tensor([alpha, 1-alpha])
        if isinstance(alpha, list): self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim()>2:
            input = input.view(input.size(0), input.size(1), -1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1, 2)    # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1, input.size(2))   # N,H*W,C => N*H*W,C
        target = target.view(-1, 1)

        logpt = F.log_softmax(input, dim=-1)
        logpt = logpt.gather(1, target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type()!=input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            select = (target!=0).type(torch.LongTensor).cuda()
            # at = self.alpha.gather(0, target.data.view(-1))
            at = self.alpha.gather(0,select.data.view(-1))
            logpt = logpt * Variable(at)

        loss = -1 * (1-pt)**self.gamma * logpt
        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()

def predictive_entropy(predictions):
    epsilon = sys.float_info.min
    pe = -np.sum(np.mean(predictions, axis=0) * np.log(np.mean(predictions, axis=0) + epsilon),
                                 axis=-1)
    return pe



def infer_dkl_v2(test_loader, model, likelihood, total_patches, n_samples = 100):
    model.eval()
    likelihood.eval()
    samples = n_samples
    lower_1c, upper_1c, mean_1, y_pred_list, y_test, probs, ftrs,  entropy_l, epi_cert, pred_var  = [], [], [], [], [], [], [], [], [], []
    st2 = time.time()
    with torch.no_grad(), gpytorch.settings.num_likelihood_samples(samples): 
        for data, target in test_loader:
            if torch.cuda.is_available():
                data, target = data.cuda(), target.cuda()
            ftr = model.feature_extractor(data)
            pred = likelihood(model(data)).probs
            ftrs.append(list(ftr.detach().cpu().numpy()))

            entropy_l.append(list(predictive_entropy(pred.cpu().numpy())))
            probs.append(list(pred.mean(0).detach().cpu().numpy()))
            pred_var.append(list(pred.var(0).detach().cpu().numpy())) # added later
            y_pred_list.append(list(pred.mean(0).argmax(-1).detach().cpu().numpy()))
            y_test.append(list(target.cpu().numpy()))

            m_1, s_1 = pred[:, :, 1].mean(0), pred[:, :, 1].std(0)
            lower_1, upper_1 = m_1-(s_1*1.96)/np.sqrt(samples), m_1+(s_1*1.96)/np.sqrt(samples)
            mean_1.append(list(m_1.detach().cpu().numpy()))
            lower_1c.append(list(lower_1.cpu().numpy()))
            upper_1c.append(list(upper_1.cpu().numpy()))

            certain = s_1**2
            epi_cert.append(list(certain.detach().cpu().numpy()))

    seconds = time.time() - st2
    minutes = seconds/60
    print(f"Time consumed in inference {minutes:.2f} minutes.\n")
    print("Throughput: {:.4f}  patches/seconds".format(total_patches/seconds))

    y_pred_list = convert_batch_list(y_pred_list)
    y_test = convert_batch_list(y_test)
    probs = convert_batch_list(probs)
    mean_1 = convert_batch_list(mean_1)
    epi_cert = convert_batch_list(epi_cert)
    pred_var = convert_batch_list(pred_var)
    lower_1c = convert_batch_list(lower_1c)
    upper_1c = convert_batch_list(upper_1c)
    ftrs = convert_batch_list(ftrs)
    entropy_l = convert_batch_list(entropy_l)

    return y_pred_list, y_test, probs, mean_1, epi_cert, pred_var, lower_1c, upper_1c, ftrs, entropy_l


# def infer_cnn(test_loader, model, total_patches, samples=1000):
#     model.eval()
#     for module in model.modules():
#         if module.__class__.__name__.startswith('Dropout'):
#             module.train()

#     y_pred, y_true, probs, feature, lower_1c, upper_1c, mean_1, epsit, pred_var = [],[], [], [], [], [], [], [], [], [], []
#     for data, target in test_loader:
#         temp_p = []
#         # for data, target in val_loader:
#         if torch.cuda.is_available():
#             data, target = data.cuda(), target.cuda()

#         for i in range(samples): # Number of monte carlo simulations
#             output, ftr = model(data)
#             un, preds = torch.max(output, 1)
#             probabilities = F.softmax(output, dim=1).detach().cpu().numpy()
#             temp_p.append(probabilities)
#         temp_p = np.array(temp_p)
#         m_1, s_1 = temp_p[:, :, 1].mean(0), temp_p[:, :, 1].std(0)
#         lower_1, upper_1 = m_1-(s_1*1.96)/np.sqrt(5), m_1+(s_1*1.96)/np.sqrt(5)

#         certain = s_1**2
#         epsit.append(list(certain.detach().cpu().numpy()))
#         pred_var.append(list(temp_p[:, :, 1].var(0).detach().cpu().numpy()))
# #
#         mean_1.append(list(m_1))
#         lower_1c.append(list(lower_1))
#         upper_1c.append(list(upper_1))
#         probs.append(list(probabilities))
#         y_pred.append(list(preds.cpu().numpy()))
#         y_true.append(list(target.cpu().numpy()))
#         feature.append(list(ftr.detach().cpu().numpy()))
#     return y_pred, y_true, probs, mean_1, epsit, pred_var, lower_1c, upper_1c, feature

def infer_cnn_v2(test_loader, model, total_patches, n_samples=1000):
    st2 = time.time()
    model.eval()
    for module in model.modules():
        if module.__class__.__name__.startswith('Dropout'):
            module.train()

    y_pred, y_true, probs,  lower_1c, upper_1c, mean_1, epsit, pred_var = [],[], [], [], [], [], [], []
    for data, target in test_loader:
        temp_p = []
        # for data, target in val_loader:
        if torch.cuda.is_available():
            data, target = data.cuda(), target.cuda()

        for i in range(n_samples): # Number of monte carlo simulations
            try:
                output, _ = model(data)
            except:
                output = model(data) 

            try:
                _, preds = torch.max(output, 1)
            except: 
                _, preds = torch.max(output[0], 1)

            try:
                probabilities = F.softmax(output, dim=1).detach().cpu().numpy()
            except: 
                probabilities = F.softmax(output[0], dim=1).detach().cpu().numpy()

            temp_p.append(probabilities)

        temp_p = np.array(temp_p)
        m_1, s_1 = temp_p[:, :, 1].mean(0), temp_p[:, :, 1].std(0)
        lower_1, upper_1 = m_1-(s_1*1.96)/np.sqrt(5), m_1+(s_1*1.96)/np.sqrt(5)

        certain = s_1**2
        epsit.append(list(certain))
        pred_var.append(list(temp_p[:, :, 1].var(0)))

        mean_1.append(list(m_1))
        lower_1c.append(list(lower_1))
        upper_1c.append(list(upper_1))
        probs.append(list(probabilities))
        y_pred.append(list(preds.cpu().numpy()))
        y_true.append(list(target.cpu().numpy()))

    seconds = time.time() - st2
    minutes = seconds/60
    print(f"Time consumed in inference in {minutes:.2f} minutes.\n")
    print("Throughput: {:.4f}  patches/seconds".format(total_patches/seconds))

    y_pred = convert_batch_list(y_pred)
    y_true = convert_batch_list(y_true)
    probs  = convert_batch_list(probs)
    mean_1 = convert_batch_list(mean_1)
    epsit  = convert_batch_list(epsit)
    pred_var = convert_batch_list(pred_var)
    lower_1c = convert_batch_list(lower_1c)
    upper_1c = convert_batch_list(upper_1c)

    return y_pred, y_true, probs, mean_1, epsit, pred_var, lower_1c, upper_1c

def count_flops(model, input_size):
    input = Variable(torch.rand(input_size))
    flops = 0
    for module in model.modules():
        if isinstance(module, torch.nn.Conv2d):
            out_h = int((input_size[2] + 2 * module.padding[0] - module.kernel_size[0]) / module.stride[0] + 1)
            out_w = int((input_size[3] + 2 * module.padding[1] - module.kernel_size[1]) / module.stride[1] + 1)
            flops += module.in_channels * module.out_channels * module.kernel_size[0] * module.kernel_size[1] * out_h * out_w / module.groups
            input_size = (input_size[0], module.out_channels, out_h, out_w)
        elif isinstance(module, torch.nn.Linear):
            flops += module.in_features * module.out_features
            input_size = (input_size[0], module.out_features)
    return flops

def make_pretty_cm(cm, categories=None, figsize=(20,20), title=None):
    """ used for saving confusion matrix in all training scripts

    """
    cm_sum = np.sum(cm, axis=1, keepdims=True)
    cm_perc = cm / cm_sum.astype(float) * 100
    annot = np.empty_like(cm).astype(str)
    nrows, ncols = cm.shape
    for i in range(nrows):
        for j in range(ncols):
            c = cm[i, j]
            p = cm_perc[i, j]
            if i == j:
                s = cm_sum[i]
                # annot[i, j] = '%.1f%%\n%d/%d' % (p, c, s)
                annot[i, j] = '%.2f%%\n%d' % (p, c)
            elif c == 0:
                annot[i, j] = '%.1f%%\n%d' % (0.0,0)
            else:
                annot[i, j] = '%.2f%%\n%d' % (p, c)

    cm = pd.DataFrame(cm, index=categories, columns=categories)
    cm.index.name = 'Actual'
    cm.columns.name = 'Predicted'

    vmin = np.min(cm)
    vmax = np.max(cm)
    off_diag_mask = np.eye(*cm.shape, dtype=bool)

    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    fig, ax = plt.subplots(figsize=figsize)
    sns.set(font_scale=2.0)
    plt.save(f"CM_{title}.png")
  