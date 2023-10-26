""" Authored by: Neel Kanwal (neel.kanwal0@gmail.com)"""
# This python file create (last) uncertainity plot shown in the paper, when DKL model is used in practice over other artifacts.
# Give paths to results when DKL model applied to different artifacts.

import matplotlib.pyplot as plt
import pandas as pd

blur_results =  "E:\\New folder1\\dkl_predictions_for_blur.xlsx"
fold_results = "E:\\New folder1\\dkl_predictions_for_fold.xlsx"

font = {'family' : 'serif',
        'weight':'normal',
        'size'   : 28}
plt.rc('font', **font)

plt.figure()
f, ax1 = plt.subplots(1, 1, figsize=(21, 9))

df = pd.read_excel(blur_results, engine='openpyxl')
df_out = df.loc[df['ground_truth'] == 0]
pred_mean = df_out['mean1'].tolist()
sigma = df_out['std1'].tolist()
sigma2= [a**2 for a in sigma]
ax1.scatter(pred_mean, sigma2, alpha=0.7, c='b', s=120, label='Artifact-free')

df_out = df.loc[df['ground_truth'] == 1]
pred_mean = df_out['mean1'].tolist()
sigma = df_out['std1'].tolist()
sigma2= [a**2 for a in sigma]
ax1.scatter(pred_mean, sigma2, alpha=0.9,  c='g', s=150, label='Blur')

df = pd.read_excel(fold_results, engine='openpyxl')
df_out = df.loc[df['ground_truth'] == 1]
pred_mean = df_out['mean1'].tolist()
sigma = df_out['std1'].tolist()
sigma2= [a**2 for a in sigma]
ax1.scatter(pred_mean, sigma2, alpha=0.9,  c='r', s=150, label='Fold')
#
# df_out = df.loc[df['ground_truth'] == 0]
# pred_mean = df_out['mean1'].tolist()
# sigma = df_out['std1'].tolist()
# sigma2= [a**2 for a in sigma]
# ax1.scatter(pred_mean, sigma2, alpha=0.7,  c='b', s=50, label='Artifact-free')

ax1.set_xlabel("Predictive Mean ($\hatp_*$)")
ax1.set_ylabel("Predictive Epistemic\nUncertainity ($\hat\sigma_{* ep}^{ 2}$)")
ax1.set_title("Blur DKL Models over Unseen Data")
ax1.set_xticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
ax1.legend(facecolor="yellow", framealpha=0.3)
plt.savefig('artifact_detection_inpractice.png')
plt.show()