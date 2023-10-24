""" Authored by: Neel Kanwal (neel.kanwal0@gmail.com)"""

# This file create a plot and provides figures from prediction sheets of DKL and Baseline predictions over FocusPath.
# update paths to the excels sheets before running.

import os
import json
import matplotlib.pyplot as plt
import pandas as pd
import shutil
import ast
import matplotlib.pyplot as plt
import numpy as np
from math import pi

df_dkl = pd.read_excel("path_to/focuspath/dkl_predictions_on_focuspath_for_blur.xlsx", engine='openpyxl')
df_dkl.rename(columns={'prediction': 'DKL_prediction'}, inplace=True)

df_baseline = pd.read_excel("path_to/focuspath/cnn_predictions_on_focuspath_for_blur.xlsx", engine='openpyxl')
df_baseline.rename(columns={'prediction': 'CNN_prediction'}, inplace=True)

label_sheet = pd.read_excel("path_to/focuspath/DatabaseInfo.xlsx", engine='openpyxl')
label_sheet.rename(columns={'Subjective Score': 'Subjective_score'}, inplace=True)

new_df = pd.concat([label_sheet[['Name', 'Subjective_score']], df_baseline[['files','CNN_prediction']], df_dkl['DKL_prediction']], axis=1)
new_df = new_df[:-2]

print(new_df.head(10))

print("Total patches in the dataset are ", len(new_df))

print("---------------------------------------------------------")
dist1, dist2, acc_cnn, acc_dkl = [], [], [], []

total_0 = len(new_df.query("Subjective_score == 0")['CNN_prediction'])
total_0_blur = new_df.query("Subjective_score == 0")['CNN_prediction'].sum()
print("Accuracy for label 0 is ", (total_0_blur/total_0)*100)

total_1 = len(new_df.query("Subjective_score == 1")['CNN_prediction']) + len(new_df.query("Subjective_score == -1")['CNN_prediction'])
total_1_blur = new_df.query("Subjective_score == 1")['CNN_prediction'].sum() + new_df.query("Subjective_score == -1")['CNN_prediction'].sum()
print("Accuracy for label 1 is ", (total_1_blur/total_1)*100)

total_2 = len(new_df.query("Subjective_score == 2")['CNN_prediction']) + len(new_df.query("Subjective_score == -2")['CNN_prediction'])
total_2_blur = new_df.query("Subjective_score == 2")['CNN_prediction'].sum() + new_df.query("Subjective_score == -2")['CNN_prediction'].sum()
print("Accuracy for label 2 is ", (total_2_blur/total_2)*100)

total_3 = len(new_df.query("Subjective_score == 3")['CNN_prediction']) + len(new_df.query("Subjective_score == -3")['CNN_prediction'])
total_3_blur = new_df.query("Subjective_score == 3")['CNN_prediction'].sum() + new_df.query("Subjective_score == -3")['CNN_prediction'].sum()
print("Accuracy for label 3 is ", (total_3_blur/total_3)*100)

total_4 = len(new_df.query("Subjective_score == 4")['CNN_prediction']) + len(new_df.query("Subjective_score == -4")['CNN_prediction'])
total_4_blur = new_df.query("Subjective_score == 4")['CNN_prediction'].sum() + new_df.query("Subjective_score == -4")['CNN_prediction'].sum()
print("Accuracy for label 4 is ", (total_4_blur/total_4)*100)

total_5 = len(new_df.query("Subjective_score == 5")['CNN_prediction']) + len(new_df.query("Subjective_score == -5")['CNN_prediction'])
total_5_blur = new_df.query("Subjective_score == 5")['CNN_prediction'].sum() + new_df.query("Subjective_score == -5")['CNN_prediction'].sum()
print("Accuracy for label 5 is ", (total_5_blur/total_5)*100)

total_6 = len(new_df.query("Subjective_score == 6")['CNN_prediction']) + len(new_df.query("Subjective_score == -6")['CNN_prediction'])
total_6_blur = new_df.query("Subjective_score == 6")['CNN_prediction'].sum() + new_df.query("Subjective_score == -6")['CNN_prediction'].sum()
print("Accuracy for label 6 is ", (total_6_blur/total_6)*100)

total_7 = len(new_df.query("Subjective_score == 7")['CNN_prediction']) + len(new_df.query("Subjective_score == -7")['CNN_prediction'])
total_7_blur = new_df.query("Subjective_score == 7")['CNN_prediction'].sum() + new_df.query("Subjective_score == -7")['CNN_prediction'].sum()
print("Accuracy for label 7 is ", (total_7_blur/total_7)*100)

total_8 = len(new_df.query("Subjective_score == 8")['CNN_prediction']) + len(new_df.query("Subjective_score == -8")['CNN_prediction'])
total_8_blur = new_df.query("Subjective_score == 8")['CNN_prediction'].sum() + new_df.query("Subjective_score == -8")['CNN_prediction'].sum()
print("Accuracy for label 8 is ", (total_8_blur/total_8)*100)

total_9 = len(new_df.query("Subjective_score == 9")['CNN_prediction']) + len(new_df.query("Subjective_score == -9")['CNN_prediction'])
total_9_blur = new_df.query("Subjective_score == 9")['CNN_prediction'].sum() + new_df.query("Subjective_score == -9")['CNN_prediction'].sum()
print("Accuracy for label 9 is ", (total_9_blur/total_9)*100)

total_10 = len(new_df.query("Subjective_score == 10")['CNN_prediction']) + len(new_df.query("Subjective_score == -10")['CNN_prediction'])
total_10_blur = new_df.query("Subjective_score == 10")['CNN_prediction'].sum() + new_df.query("Subjective_score == -10")['CNN_prediction'].sum()
print("Accuracy for label 10 is ", (total_10_blur/total_10)*100)

total_11 = len(new_df.query("Subjective_score == 11")['CNN_prediction']) + len(new_df.query("Subjective_score == -11")['CNN_prediction'])
total_11_blur = new_df.query("Subjective_score == 11")['CNN_prediction'].sum() + new_df.query("Subjective_score == -11")['CNN_prediction'].sum()
print("Accuracy for label 11 is ", (total_11_blur/total_11)*100)

total_12 = len(new_df.query("Subjective_score == 12")['CNN_prediction']) + len(new_df.query("Subjective_score == -12")['CNN_prediction'])
total_12_blur = new_df.query("Subjective_score == 12")['CNN_prediction'].sum() + new_df.query("Subjective_score == -12")['CNN_prediction'].sum()
print("Accuracy for label 12 is ", (total_12_blur/total_12)*100)

total_13 = len(new_df.query("Subjective_score == 13")['CNN_prediction']) + len(new_df.query("Subjective_score == -13")['CNN_prediction']) + len(new_df.query("Subjective_score == -14")['CNN_prediction']) + len(new_df.query("Subjective_score == 14")['CNN_prediction'])
total_13_blur = new_df.query("Subjective_score == 13")['CNN_prediction'].sum() + new_df.query("Subjective_score == -13")['CNN_prediction'].sum() + new_df.query("Subjective_score == -14")['CNN_prediction'].sum() + new_df.query("Subjective_score == 14")['CNN_prediction'].sum()
print("Accuracy for label 13 is ", (total_13_blur/total_13)*100)

dist1.extend([total_0 ,total_1 ,total_2 ,total_3 ,total_4 ,total_5 ,total_6 ,total_7 ,total_8 ,total_9 ,total_10 ,total_11 ,total_12 ,total_13])
acc_cnn.extend([total_0_blur ,total_1_blur ,total_2_blur ,total_3_blur ,total_4_blur ,total_5_blur ,total_6_blur ,total_7_blur ,total_8_blur ,total_9_blur ,total_10_blur ,total_11_blur ,total_12_blur ,total_13_blur])

tot = total_0+total_1+total_2+total_3+total_4+total_5+total_6+total_7+total_8+total_9+total_10+total_11+total_12+total_13

if len(new_df) == tot:
    print("Length matches for CNN")

print("-----------------------------------------")
total_0 = len(new_df.query("Subjective_score == 0")['DKL_prediction'])
total_0_blur = new_df.query("Subjective_score == 0")['DKL_prediction'].sum()
print("Accuracy for label 0 is ", (total_0_blur/total_0)*100)

total_1 = len(new_df.query("Subjective_score == 1")['DKL_prediction']) + len(new_df.query("Subjective_score == -1")['DKL_prediction'])
total_1_blur = new_df.query("Subjective_score == 1")['DKL_prediction'].sum() + new_df.query("Subjective_score == -1")['DKL_prediction'].sum()
print("Accuracy for label 1 is ", (total_1_blur/total_1)*100)

total_2 = len(new_df.query("Subjective_score == 2")['DKL_prediction']) + len(new_df.query("Subjective_score == -2")['DKL_prediction'])
total_2_blur = new_df.query("Subjective_score == 2")['DKL_prediction'].sum() + new_df.query("Subjective_score == -2")['DKL_prediction'].sum()
print("Accuracy for label 2 is ", (total_2_blur/total_2)*100)

total_3 = len(new_df.query("Subjective_score == 3")['DKL_prediction']) + len(new_df.query("Subjective_score == -3")['DKL_prediction'])
total_3_blur = new_df.query("Subjective_score == 3")['DKL_prediction'].sum() + new_df.query("Subjective_score == -3")['DKL_prediction'].sum()
print("Accuracy for label 3 is ", (total_3_blur/total_3)*100)

total_4 = len(new_df.query("Subjective_score == 4")['DKL_prediction']) + len(new_df.query("Subjective_score == -4")['DKL_prediction'])
total_4_blur = new_df.query("Subjective_score == 4")['DKL_prediction'].sum() + new_df.query("Subjective_score == -4")['DKL_prediction'].sum()
print("Accuracy for label 4 is ", (total_4_blur/total_4)*100)

total_5 = len(new_df.query("Subjective_score == 5")['DKL_prediction']) + len(new_df.query("Subjective_score == -5")['DKL_prediction'])
total_5_blur = new_df.query("Subjective_score == 5")['DKL_prediction'].sum() + new_df.query("Subjective_score == -5")['DKL_prediction'].sum()
print("Accuracy for label 5 is ", (total_5_blur/total_5)*100)

total_6 = len(new_df.query("Subjective_score == 6")['DKL_prediction']) + len(new_df.query("Subjective_score == -6")['DKL_prediction'])
total_6_blur = new_df.query("Subjective_score == 6")['DKL_prediction'].sum() + new_df.query("Subjective_score == -6")['DKL_prediction'].sum()
print("Accuracy for label 6 is ", (total_6_blur/total_6)*100)

total_7 = len(new_df.query("Subjective_score == 7")['DKL_prediction']) + len(new_df.query("Subjective_score == -7")['DKL_prediction'])
total_7_blur = new_df.query("Subjective_score == 7")['DKL_prediction'].sum() + new_df.query("Subjective_score == -7")['DKL_prediction'].sum()
print("Accuracy for label 7 is ", (total_7_blur/total_7)*100)

total_8 = len(new_df.query("Subjective_score == 8")['DKL_prediction']) + len(new_df.query("Subjective_score == -8")['DKL_prediction'])
total_8_blur = new_df.query("Subjective_score == 8")['DKL_prediction'].sum() + new_df.query("Subjective_score == -8")['DKL_prediction'].sum()
print("Accuracy for label 8 is ", (total_8_blur/total_8)*100)

total_9 = len(new_df.query("Subjective_score == 9")['DKL_prediction']) + len(new_df.query("Subjective_score == -9")['DKL_prediction'])
total_9_blur = new_df.query("Subjective_score == 9")['DKL_prediction'].sum() + new_df.query("Subjective_score == -9")['DKL_prediction'].sum()
print("Accuracy for label 9 is ", (total_9_blur/total_9)*100)

total_10 = len(new_df.query("Subjective_score == 10")['DKL_prediction']) + len(new_df.query("Subjective_score == -10")['DKL_prediction'])
total_10_blur = new_df.query("Subjective_score == 10")['DKL_prediction'].sum() + new_df.query("Subjective_score == -10")['DKL_prediction'].sum()
print("Accuracy for label 10 is ", (total_10_blur/total_10)*100)

total_11 = len(new_df.query("Subjective_score == 11")['DKL_prediction']) + len(new_df.query("Subjective_score == -11")['DKL_prediction'])
total_11_blur = new_df.query("Subjective_score == 11")['DKL_prediction'].sum() + new_df.query("Subjective_score == -11")['DKL_prediction'].sum()
print("Accuracy for label 11 is ", (total_11_blur/total_11)*100)

total_12 = len(new_df.query("Subjective_score == 12")['DKL_prediction']) + len(new_df.query("Subjective_score == -12")['DKL_prediction'])
total_12_blur = new_df.query("Subjective_score == 12")['DKL_prediction'].sum() + new_df.query("Subjective_score == -12")['DKL_prediction'].sum()
print("Accuracy for label 12 is ", (total_12_blur/total_12)*100)

total_13 = len(new_df.query("Subjective_score == 13")['DKL_prediction']) + len(new_df.query("Subjective_score == -13")['DKL_prediction']) + len(new_df.query("Subjective_score == -14")['DKL_prediction']) + len(new_df.query("Subjective_score == 14")['DKL_prediction'])
total_13_blur = new_df.query("Subjective_score == 13")['DKL_prediction'].sum() + new_df.query("Subjective_score == -13")['DKL_prediction'].sum() + new_df.query("Subjective_score == -14")['DKL_prediction'].sum() + new_df.query("Subjective_score == 14")['DKL_prediction'].sum()
print("Accuracy for label 13 is ", (total_13_blur/total_13)*100)

tot = total_0+total_1+total_2+total_3+total_4+total_5+total_6+total_7+total_8+total_9+total_10+total_11+total_12+total_13
if len(new_df) == tot:
    print("Length matches for DKL")
dist2.extend([total_0 ,total_1 ,total_2 ,total_3 ,total_4 ,total_5 ,total_6 ,total_7 ,total_8 ,total_9 ,total_10 ,total_11 ,total_12 ,total_13])
acc_dkl.extend([total_0_blur ,total_1_blur ,total_2_blur ,total_3_blur ,total_4_blur ,total_5_blur ,total_6_blur ,total_7_blur ,total_8_blur ,total_9_blur ,total_10_blur ,total_11_blur ,total_12_blur ,total_13_blur])
print("--------------------------------------------------------")

print("Distribution of Samples ", dist1)
print("Distribution of Samples ", dist2)
print("CNN Accuracy ", acc_cnn)
print("DKL Accuracy", acc_dkl )

dist = dist1
# bar_width = 0.5
# x = list(range(0, len(dist)))
# plt.bar(x, dist, color ='r', width = bar_width,  edgecolor ='black', label ='Total Samples')
# plt.bar(x, acc_cnn, color ='g', width = bar_width, edgecolor ='black', label ='Detected blur by CNN')
# plt.bar(x, acc_dkl, color ='b', width = bar_width, edgecolor ='black', label ='Detected blur by DKL')
#
# plt.xlabel('Ordinal Blur Labels', fontsize=12)
# plt.ylabel('No. of Samples', fontsize=12)
# plt.title("Inference results on FOCUSPATH")
# plt.xticks(x)
# plt.legend()
# # plt.show()
# plt.savefig(f"FOCUSPATH inference.png")

fig = plt.figure(figsize=(15, 8))
bar_width = 0.25
x = np.array(list(range(0, len(dist1))))
plt.bar(x-0.3, dist1, width=bar_width, color='r', align='center',  edgecolor ='black', label ='Total Samples')
plt.bar(x, acc_cnn, width=bar_width, color='g', align='center', edgecolor ='black', label ='Detected blur by CNN')
plt.bar(x+0.3, acc_dkl, width=bar_width, color='b', align='center', edgecolor ='black', label ='Detected blur by DKL')
plt.xlabel('$Ordinal\ Blur\ Labels$', fontsize=12)
plt.ylabel('$No.\ of\ Samples$', fontsize=12)
plt.xticks(x)
plt.title("$Inference\ on\ FOCUSPATH$")
plt.legend()
plt.savefig(f"FOCUSPATH inference.png")


plt.gcf().set_size_inches(12, 12)
# SpiderPlot
gp = pd.DataFrame({"group": ['Total Samples', 'Detected Blur by CNN', 'Detected Blur by DKL']})
df = pd.DataFrame(np.array([dist, acc_cnn, acc_dkl]))
output_df = pd.concat([gp,df], axis=1)

# number of variable
labels =list(output_df)[1:]
N = len(labels)
 
# What will be the angle of each axis in the plot? (we divide the plot / number of variable)
angles = [n / float(N) * 2 * pi for n in range(N)]
angles += angles[:1]
 
# Initialise the spider plot
ax = plt.subplot(111, polar=True)
 
# If you want the first axis to be on top:
ax.set_theta_offset(pi / 2)
ax.set_theta_direction(-1)
 
# Draw one axe per variable + add labels
plt.xticks(angles[:-1], labels)
 
plt.yticks(list(np.arange(0, 1100, 100)), color="grey", size=9)
plt.ylim(0,1100)

# Ind1
values = output_df.loc[0].drop('group').values.flatten().tolist()
values += values[:1]
ax.plot(angles, values, linewidth=1, linestyle='solid', label="Total Samples")
ax.fill(angles, values, 'b', alpha=0.1)
 
# Ind2
values = output_df.loc[1].drop('group').values.flatten().tolist()
values += values[:1]
ax.plot(angles, values, linewidth=1, linestyle='solid', label="Detected blur by CNN")
ax.fill(angles, values, 'r', alpha=0.1)

# Ind3
values = output_df.loc[2].drop('group').values.flatten().tolist()
values += values[:1]
ax.plot(angles, values, linewidth=1, linestyle='solid', label="Detected blur by DKL")
ax.fill(angles, values, 'r', alpha=0.1)

# Add legend
plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
plt.savefig(f"FOCUSPATH RadarPlot.png")


# Show the graph
# plt.show()
print("Data merged and saved as xlsx")
new_df.to_excel("path_to/combined_focuspath_results.xlsx", index=False, header=True)