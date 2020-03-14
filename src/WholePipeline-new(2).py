# -*- coding: utf-8 -*-
"""
Created on Sun Mar  8 10:40:52 2020

@author: pc
"""

import csv
import numpy as np
import os
from src import anogan

from keras.models import load_model
from keras.preprocessing import image
from keras.preprocessing.image import img_to_array, load_img
from sklearn.metrics import confusion_matrix
from matplotlib import pyplot

inception = load_model("../NoisyVSTissueModel.h5")

# validation_datagen = ImageDataGenerator(rescale=1./255)

image_size = 299

# batchsize = 128
# nb_epoch = 20


def mse(image_a, image_b):
    error = np.sum((image_a.astype("float") - image_b.astype("float")) ** 2)
    error /= float(image_a.shape[0] * image_a.shape[1])

    return error


# rootdir = './test images'
outputList = []

# new code

# define the path to the test data folders:
# Positives are the Anomaly test images. They are in a foldr named "./Test data/positives
# Negatives are the Normal test images. They are in a folder named "./Test data/HealthyAll
testDataDir = "./Test data"
pathTestDataPositives = testDataDir + '/' + "positives"
pathTestDataNegatives = testDataDir + '/' + "HealthyAll"

# find the total number of Positives and Negatives in the test data

totalPositives = len([name for name in os.listdir(pathTestDataPositives) if os.path.isfile(name)])
totalNegatives = len([name for name in os.listdir(pathTestDataNegatives) if os.path.isfile(name)])
# create a Y ground-truth (expected real test Y labels)
y_test = [0 for _ in range(totalNegatives)]  # fill 0 labele for the total number of negatives in the test set
# append the y_test with "1" labele for the total number of negatives
y_test.append([1 for _ in range(totalPositives)])

y_prediction = []  # this will contain the predicted labels of the test data

MIN_GAN_LOSS_MSE = 1  # the minimum expected GAN loss MSE
MAX_GAN_LOSS_MSE = 99  # TODO put the right value

FPR_array = []
TPR_array = []
Thresholds_array = []

# This script performs the "test" step on the model. It performs
# prediction on the test data, select the optimum threshold and saves it as the prediction threshold for prediction
# script.
#
# First step in this script runs prediction on all test data (negatives and positives) for a several threshold values
# in a loop and collects that FPR and TPR for each threshold value, and plots the ROC curve from all the TPR vs. FPR
# points. Second step in this script finds the threshold value from the previous step for which the FPR == 0.0005 and
# then finds the corresponding TPR value. The selected threshold is saved to be used as the parameter for prediction
#
model = anogan.anomaly_detector(g=None, d=None)
intermidiate_model = anogan.feature_extractor(None)
# loop through a range of thresholds, for each threshold, predict ALL test data, compute the FPR and TPR
# The order of predicted images should be in the same order of the epected Y labels (in y_prediction)
# When finished, the ROC curve is plotted from all FPR, TPR points
for thresh in range(MAX_GAN_LOSS_MSE, MIN_GAN_LOSS_MSE):  # thresholds should be in descending order
    # start with the negatives first because we put thier labels first in the y_test
    for subdir, dirs, files in os.walk(pathTestDataNegatives):
        for file in files:
            img_path = subdir + '/' + file
            print(img_path)
            patch = image.load_img(img_path, target_size=(image_size, image_size))
            x = image.img_to_array(patch) / 255
            x = np.expand_dims(x, axis=0)
            #        x = preprocess_input(x)
            preds = inception.predict(x)
            label = np.argmax(preds[0])

            # if the Inception CNN model determined that this is a Normal (negative) image,
            # save the (0) label immediately, else, let the GAN anomaly detection model to determine
            if label == 0:
                y_prediction.append(0)
            else:
                img = load_img(img_path, target_size=(128, 128, 3), grayscale=False)
                x_img = img_to_array(img)
                x_img = x_img.squeeze() / 255
                # get the score (loss) of the GAN model, and the predicted image
                score, similar_img = anogan.compute_anomaly_score(model, x_img.reshape(1, 128, 128, 3),
                                                                  intermidiate_model, iterations=100)
                # compute the Mean Squared Error of the predicted image vs. the input image
                err = mse(similar_img, x_img)

                if err < thresh and preds[0, label] > 0.95:
                    y_prediction.append(0)
                else:
                    y_prediction.append(1)

    for subdir, dirs, files in os.walk(pathTestDataPositives):
        for file in files:
            img_path = subdir + '/' + file
            print(img_path)
            patch = image.load_img(img_path, target_size=(image_size, image_size))
            x = image.img_to_array(patch) / 255
            x = np.expand_dims(x, axis=0)
            #        x = preprocess_input(x)
            preds = inception.predict(x)
            label = np.argmax(preds[0])

            if label == 0:
                y_prediction.append(0)
            else:
                img = load_img(img_path, target_size=(128, 128, 3), grayscale=False)
                x_img = img_to_array(img)
                x_img = x_img.squeeze() / 255

                score, similar_img = anogan.compute_anomaly_score(model, x_img.reshape(1, 128, 128, 3),
                                                                  intermidiate_model, iterations=100)
                err = mse(similar_img, x_img)

                if err < thresh and preds[0, label] > 0.95:
                    y_prediction.append(0)
                else:
                    y_prediction.append(1)

    # save the threshold of this iteration
    Thresholds_array.append(thresh)
    # find the confusion matrix for this threshold computation, usingthe groundtruth Y labels vs. the predicted labels
    # predicted lables are in y_prediction
    conf_matrix = confusion_matrix(y_test, y_prediction)
    fp = conf_matrix[0, 1]
    fpr = fp / totalNegatives
    FPR_array.append(fpr)
    tp = conf_matrix[1, 1]
    tpr = tp / totalPositives
    TPR_array.append(tpr)

# plot the roc curve for the model
pyplot.plot(FPR_array, TPR_array, marker='.', label='my model')
# axis labels
pyplot.xlabel('False Positive Rate')
pyplot.ylabel('True Positive Rate')
# show the legend
pyplot.legend()
# show the plot
pyplot.show()

# TODO find the threshhold for which the FPR == 0.0005 (or less), and find the corresponding TPR
# this threshold will be used later for actual prediction (below)
# start from FPR == 0 and up. the curve might bounce up and down, so TPR might go up then down
fpr = 0
tpr = 0
thresh = 0
# starting from the top of the array (the highest FPR), down
# while fpr in PFR_array, is > REQUESTED_MAX_FPR
# tpr = TPR_array[idx]
# thresh = Thresholds_array[idx]

print("Optimum Threshold for test data: ", thresh)
print("Test data FPR: ", fpr)
print("Test data TPR: ", tpr)
# TODO save the optimum threshold where another prediction script cna use it

# end new code ##############################

#####################################################
# This part of the script performs the actual final prediction on the test data, based on the optimum Threshold
# selected in the previous step
#####################################################
# old modified code ##################
# thresh = 11
model = anogan.anomaly_detector(g=None, d=None)
intermidiate_model = anogan.feature_extractor(None)
# Generate heatmaps for feature extraction of the last concatenate layer      
for subdir, dirs, files in os.walk(testDataDir):
    for file in files:
        img_path = subdir + '/' + file
        print(img_path)
        #        img_path =
        patch = image.load_img(img_path, target_size=(image_size, image_size))
        x = image.img_to_array(patch) / 255
        x = np.expand_dims(x, axis=0)
        #        x = preprocess_input(x)
        preds = inception.predict(x)
        label = np.argmax(preds[0])

        if label == 0:
            #            outputList.append([file,'Normal'])
            outputList.append([file, thresh - 1, 'Normal'])
        else:
            img = load_img(img_path, target_size=(128, 128, 3), grayscale=False)
            x_img = img_to_array(img)
            x_img = x_img.squeeze() / 255

            score, similar_img = anogan.compute_anomaly_score(model, x_img.reshape(1, 128, 128, 3), intermidiate_model,
                                                              iterations=100)
            err = mse(similar_img, x_img)

            if err < thresh and preds[0, label] > 0.95:
                #                outputList.append([file,'Normal'])
                outputList.append([file, err, 'Normal'])
            else:
                #                outputList.append([file,'Anomaly'])
                outputList.append([file, err, 'Anomaly'])

with open('results.csv', 'w', newline='') as myfile:
    wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
    wr.writerow(outputList)


if __name__ == '__main__':
    pass
