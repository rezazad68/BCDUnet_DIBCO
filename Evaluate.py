# -*- coding: utf-8 -*-
"""
Created on Sat Jun  8 18:15:43 2019

@author: Reza winchester
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import models as M
import numpy as np
import utils as UT
import matplotlib.pyplot as plt
#scikit learn
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import jaccard_similarity_score
from sklearn.metrics import f1_score

## Parameters
batch_size  = 8
patch_h     = 128
patch_w     = 128
stride_h    = 5
stride_w    = 5
Batch_size  = 8


Estimated_masks = []
True_masks      = []

## Get the dataset for test
Dataset_add = './DIBCO/' 
Te_d, Te_m = UT.get_test_data(Dataset_add, 2016)

## Model
model = M.BCDU_net_D3(input_size = (128, 128,1))
model.load_weights('weight_text.hdf5')

y_scores = []
y_true   = []

## Mask estimation using batches
for idx in range(len(Te_d)):
    IMG = Te_d[idx]
    MSK = Te_m[idx]
    patches , new_h, new_w = UT.extract_ordered_overlap(IMG, patch_h, patch_w, stride_h, stride_w)
    patches     = np.expand_dims(patches,  axis = 3)
    predictions = model.predict(patches, batch_size= Batch_size, verbose=1)
    estimated   = UT.recompone_overlap(predictions[:,:,:,0], new_h, new_w, stride_h, stride_w)
    estimated   = np.where(estimated >= 0.5, 1, 0)
    MSK         = MSK[0: new_h, 0:new_w]
    MSK         = np.where(MSK >= 0.5, 1, 0)
    estimated   = estimated.reshape(estimated.shape[0]*estimated.shape[1], 1)
    MSK         = MSK.reshape(MSK.shape[0]*MSK.shape[1], 1)

    y_scores    = np.concatenate((y_scores, estimated), axis=None)
    y_true      = np.concatenate((y_true,   MSK)      , axis=None)


##################### Measurements ############################3

output_folder = 'output/'

#Area under the ROC curve
fpr, tpr, thresholds = roc_curve((y_true), y_scores)
AUC_ROC = roc_auc_score(y_true, y_scores)
print ("\nArea under the ROC curve: " +str(AUC_ROC))
roc_curve =plt.figure()
plt.plot(fpr,tpr,'-',label='Area Under the Curve (AUC = %0.4f)' % AUC_ROC)
plt.title('ROC curve')
plt.xlabel("FPR (False Positive Rate)")
plt.ylabel("TPR (True Positive Rate)")
plt.legend(loc="lower right")
plt.savefig(output_folder+"ROC.png")

#Precision-recall curve
precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
precision = np.fliplr([precision])[0] 
recall = np.fliplr([recall])[0]
AUC_prec_rec = np.trapz(precision,recall)
print ("\nArea under Precision-Recall curve: " +str(AUC_prec_rec))
prec_rec_curve = plt.figure()
plt.plot(recall,precision,'-',label='Area Under the Curve (AUC = %0.4f)' % AUC_prec_rec)
plt.title('Precision - Recall curve')
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.legend(loc="lower right")
plt.savefig(output_folder+"Precision_recall.png")

#Confusion matrix
threshold_confusion = 0.5
print ("\nConfusion matrix:  Custom threshold (for positive) of " +str(threshold_confusion))
y_pred = np.empty((y_scores.shape[0]))
for i in range(y_scores.shape[0]):
    if y_scores[i]>=threshold_confusion:
        y_pred[i]=1
    else:
        y_pred[i]=0
confusion = confusion_matrix(y_true, y_pred)
print (confusion)
accuracy = 0
if float(np.sum(confusion))!=0:
    accuracy = float(confusion[0,0]+confusion[1,1])/float(np.sum(confusion))
print ("Global Accuracy: " +str(accuracy))
specificity = 0
if float(confusion[0,0]+confusion[0,1])!=0:
    specificity = float(confusion[0,0])/float(confusion[0,0]+confusion[0,1])
print ("Specificity: " +str(specificity))
sensitivity = 0
if float(confusion[1,1]+confusion[1,0])!=0:
    sensitivity = float(confusion[1,1])/float(confusion[1,1]+confusion[1,0])
print ("Sensitivity: " +str(sensitivity))
precision = 0
if float(confusion[1,1]+confusion[0,1])!=0:
    precision = float(confusion[1,1])/float(confusion[1,1]+confusion[0,1])
print ("Precision: " +str(precision))

#Jaccard similarity index
jaccard_index = jaccard_similarity_score(y_true, y_pred, normalize=True)
print ("\nJaccard similarity score: " +str(jaccard_index))

#F1 score
F1_score = f1_score(y_true, y_pred, labels=None, average='binary', sample_weight=None)
print ("\nF1 score (F-measure): " +str(F1_score))

#Save the results
file_perf = open(output_folder+'performances.txt', 'w')
file_perf.write("Area under the ROC curve: "+str(AUC_ROC)
                + "\nArea under Precision-Recall curve: " +str(AUC_prec_rec)
                + "\nJaccard similarity score: " +str(jaccard_index)
                + "\nF1 score (F-measure): " +str(F1_score)
                +"\n\nConfusion matrix:"
                +str(confusion)
                +"\nACCURACY: " +str(accuracy)
                +"\nSENSITIVITY: " +str(sensitivity)
                +"\nSPECIFICITY: " +str(specificity)
                +"\nPRECISION: " +str(precision)
                )
file_perf.close()
