""" Authored by: Neel Kanwal (neel.kanwal0@gmail.com)"""

# This file provides confidence plots from the predicted xlxs sheets of DKL and Baseline Models.
# This scrip requires running merging prediction xlxs sheets first. That will combines results in one sheet to make a single plot as shown in the paper.
# Update path to excel sheets for running this script.

import os
import json
import matplotlib.pyplot as plt
import pandas as pd
import shutil
import ast
from PIL import Image

## Combining prediction sheets
path_to_dkl_results = "path_to/dkl_predictions_val.xlsx"
path_to_baseline_results = "path_to/dkl_predictions_val.xlsx"

df_dkl = pd.read_excel(path_to_dkl_results, engine='openpyxl')
df_baseline = pd.read_excel(path_to_baseline_results, engine='openpyxl')

df_dkl.rename(columns={'prediction': 'DKL_prediction', 'mean1':'dkl_mean', 'lower_conf':'dkl_lower', "upper_conf":'dkl_upper'}, inplace=True)
df_baseline.rename(columns={'prediction': 'CNN_prediction','mean1':'cnn_mean', 'lower_conf': 'cnn_lower', 'upper_conf': 'cnn_upper'}, inplace=True)


new_df = pd.concat([df_dkl[['files', 'ground_truth']], df_baseline[['CNN_prediction', 'cnn_mean', 'cnn_lower', 'cnn_upper']], df_dkl[['DKL_prediction', 'dkl_mean', 'dkl_lower', 'dkl_upper']]], axis=1)

print("Data merged and saved as xlsx")
new_df.to_excel("combined_emc_results_for_plots.xlsx", index=False, header=True)

samples = 15 # Numbers of sample to extract randomly and compare.

# Choose one option to compare catergory of predictions
take_only_tp = True
take_only_fp = False
take_only_fn = False
take_only_tn = False

fname = "TP.png" # Name to save

# df = pd.read_excel("/path_to/combined_emc_val_blur_results.xlsx", engine='openpyxl')
df = new_df
dateset_dir = "path_to/artifact_dataset/blur/validation/"

if take_only_tp:
    df_out = df.loc[(df['ground_truth'] == 1) & (df["DKL_prediction"] == 1) & (df["CNN_prediction"] == 1)].sample(n=samples)
elif take_only_fp:
    df_out = df.loc[(df['ground_truth'] == 0) & (df["DKL_prediction"] == 1)].sample(n=samples)
elif take_only_fn:
    df_out = df.loc[(df['ground_truth'] == 1) & (df["DKL_prediction"] == 0)].sample(n=samples)
elif take_only_tn:
    df_out = df.loc[(df['ground_truth'] == 0) & (df["DKL_prediction"] == 0) & (df["CNN_prediction"] == 0)].sample(n=samples)
else:
    df_out = df.sample(n=samples)

filenames = df_out['files'].tolist()
mean_cnn = df_out['cnn_mean'].tolist()
lower_cnn = df_out['cnn_lower'].tolist()
upper_cnn = df_out['cnn_upper'].tolist()

mean_dkl = df_out['dkl_mean'].tolist()
lower_dkl = df_out['dkl_lower'].tolist()
upper_dkl = df_out['dkl_upper'].tolist()

ticks = []
for val in range(len(df_out)):
    if df_out.iloc[val]['ground_truth'] == 0 and df_out.iloc[val]['DKL_prediction'] == 0:
        ticks.append("TN")
    elif df_out.iloc[val]['ground_truth'] == 0 and df_out.iloc[val]['DKL_prediction'] == 1:
        ticks.append("FP")
    elif df_out.iloc[val]['ground_truth'] == 1 and df_out.iloc[val]['DKL_prediction'] == 0:
        ticks.append("FN")
    elif df_out.iloc[val]['ground_truth'] == 1 and df_out.iloc[val]['DKL_prediction'] == 1:
        ticks.append("TP")

# This will bring-up corresponding patches to show on a seperate plots
plt.figure()
fig = plt.figure(figsize=(12, 3))
for i in range(0, samples):
    fig.add_subplot(1, samples, i+1)    # the number of images in the grid is 5*5 (25)
    plt.subplots_adjust(hspace=1)
    plt.axis("off")
    if ticks[i] == "TN":
        plt.imshow(Image.open(os.path.join(dateset_dir, "artifact_free", filenames[i])))
        plt.title(f"{i+1}")
    elif ticks[i] == "TP":
        plt.imshow(Image.open(os.path.join(dateset_dir, "blur", filenames[i])))
        plt.title(f"{i+1}")
    elif ticks[i] == "FP":
        plt.imshow(Image.open(os.path.join(dateset_dir, "artifact_free", filenames[i])))
        plt.title(f"{i+1}")
    elif ticks[i] == "FN":
        plt.imshow(Image.open(os.path.join(dateset_dir, "blur", filenames[i])))
        plt.title(f"{i+1}")
plt.xticks(list(range(samples)))
plt.savefig("Corresponding_patches.png")
# plt.show()

plt.figure()
f, ax1 = plt.subplots(1, 1, figsize=(15, 8))
x = list(range(1, samples+1))

# # Shade between the lower and upper confidence bounds
ax1.plot(x, mean_dkl, color='k', linestyle='--', marker='D', markersize=8,  linewidth=2, label="DKL Prediction")
ax1.fill_between(x, lower_dkl, upper_dkl, alpha=0.5, color='green', label= "DKL 95% Confidence")

ax1.plot(x, mean_cnn, color='r', linestyle='-', marker='s', markersize=8,  linewidth=2, label="Baseline Prediction")
ax1.fill_between(x, lower_cnn, upper_cnn, alpha=0.6, color='red', label="Baseline 95% Confidence")

ax1.set(xticks=x, xticklabels=list(range(1,samples+1)))
ax1.set_xlabel("Input Samples")
ax1.set_ylabel("Predictive Mean")
ax1.set_title("True Positive")

ax1.legend(facecolor="yellow", loc="lower right", framealpha=0.3)
plt.savefig(f"{fname}")
# plt.show()

# plt.figure()
# f, ax = plt.subplots(1, 1, figsize=(15, 6))
# x1 = [a-0.2 for a in x]
# x2 = [a+0.2 for a in x]

# ax.bar(x1, mean_dkl, width = 0.3, color='cyan', edgecolor='gray', capsize=10, label='DKL prediction for blur')
# ax.bar(x2, mean_cnn, width = 0.3, color='lightgreen', edgecolor='gray', capsize=10, label='CNN prediction for blur')

# yerr_cnn = [(b-a) for a, b in zip(lower_cnn, upper_cnn)]
# yerr = [(b-a) for a, b in zip(lower_dkl, upper_dkl)]
# ax.errorbar(x1, mean_dkl, yerr=yerr, fmt='D', ecolor='green', color='k', elinewidth=5, capsize=7, label="DKL Prediction confidence")
# ax.errorbar(x2, mean_cnn, yerr=yerr_cnn, fmt='D', ecolor='red', color='k', elinewidth=5, capsize=7, label="CNN Prediction confidence")

# ax.set(xticks=x, xticklabels=ticks)
# ax.set_xlabel("Input Samples")
# ax.set_ylabel("Predictive Mean")