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
from settings import MODEL_PATH, NEGATIVE_DIR, POSITIVE_DIR, MAX_GAN_LOSS_MSE, MIN_GAN_LOSS_MSE, LOCAL, \
    TEMP_DIR, ROC_CURVE_PATH, OPT_THRESH, OPT_THRESH_PATH, OPT_RESULT_CSV
from utils.folder_file_manager import save_file


class WholePipeline:
    """
    # Positives are the Anomaly test images. They are in a foldr named "./test_data/positives
    # Negatives are the Normal test images. They are in a folder named "./test_data/HealthyAll

    # This script performs the "test" step on the model. It performs
    # prediction on the test data, select the optimum threshold and saves it as the prediction threshold for prediction
    # script.

    # First step in this script runs prediction on all test data (negatives and positives) for a several threshold
    values in a loop and collects that FPR and TPR for each threshold value, and plots the ROC curve from all the
    TPR vs. FPR points. Second step in this script finds the threshold value from the previous step for which the
    FPR == 0.0005 and then finds the corresponding TPR value. The selected threshold is saved to be used as
    the parameter for prediction loop through a range of thresholds, for each threshold, predict ALL test data,
    compute the FPR and TPR The order of predicted images should be in the same order of the expected Y labels
    (in y_prediction) When finished, the ROC curve is plotted from all FPR, TPR points
    """

    def __init__(self):
        self.inception = load_model(MODEL_PATH)
        self.model = anogan.anomaly_detector(g=None, d=None)
        self.intermediate_model = anogan.feature_extractor(None)

        self.image_size = 299
        self.y_prediction = []  # this will contain the predicted labels of the test data
        self.y_test = []
        self.fpr_array = []
        self.tpr_array = []
        self.thresholds_array = []
        self.output_list = []

        self.total_positives = len([name for name in os.listdir(POSITIVE_DIR) if os.path.isfile(name)])
        self.total_negatives = len([name for name in os.listdir(NEGATIVE_DIR) if os.path.isfile(name)])
        self.y_test = [0] * self.total_negatives + [1] * self.total_positives

    @staticmethod
    def plot_roc_curve(f_array, t_array):

        pyplot.plot(f_array, t_array, marker='.', label='my model')
        # axis labels
        pyplot.xlabel('False Positive Rate')
        pyplot.ylabel('True Positive Rate')
        # show the legend
        pyplot.legend()
        pyplot.savefig(ROC_CURVE_PATH)
        # show the plot
        pyplot.show()

    def get_fpr_tpr(self, dir_path, thresh, opt_thresh=False):

        for subdir, dirs, files in os.walk(dir_path):
            for file in files:
                img_path = subdir + '/' + file
                print(img_path)
                patch = image.load_img(img_path, target_size=(self.image_size, self.image_size))
                x = image.img_to_array(patch) / 255
                x = np.expand_dims(x, axis=0)
                predicts = self.inception.predict(x)
                label = np.argmax(predicts[0])

                # if the Inception CNN model determined that this is a Normal (negative) image,
                # save the (0) label immediately, else, let the GAN anomaly detection model to determine
                if label == 0:
                    if opt_thresh:
                        self.output_list.append([file, thresh - 1, 'Normal'])
                    else:
                        self.y_prediction.append(0)
                else:
                    img = load_img(img_path, target_size=(128, 128, 3), grayscale=False)
                    x_img = img_to_array(img)
                    x_img = x_img.squeeze() / 255
                    # get the score (loss) of the GAN model, and the predicted image
                    score, similar_img = anogan.compute_anomaly_score(self.model, x_img.reshape(1, 128, 128, 3),
                                                                      self.intermediate_model, iterations=100)
                    # compute the Mean Squared Error of the predicted image vs. the input image
                    err = mse(similar_img, x_img)

                    if opt_thresh:

                        if err < thresh and predicts[0, label] > 0.95:
                            self.output_list.append([file, err, 'Normal'])
                        else:
                            self.output_list.append([file, err, 'Anomaly'])

                        with open(OPT_RESULT_CSV, 'w', newline='') as my_file:
                            wr = csv.writer(my_file, quoting=csv.QUOTE_ALL)
                            wr.writerow(self.output_list)

                    else:

                        if err < thresh and predicts[0, label] > 0.95:
                            self.y_prediction.append(0)
                        else:
                            self.y_prediction.append(1)

    def collect_fpr_tpr_all_thresh(self):

        for thresh_val in range(MIN_GAN_LOSS_MSE, MAX_GAN_LOSS_MSE):  # thresholds should be in descending order
            # start with the negatives first because we put thier labels first in the y_test
            self.get_fpr_tpr(dir_path=NEGATIVE_DIR, thresh=thresh_val)
            self.get_fpr_tpr(dir_path=POSITIVE_DIR, thresh=thresh_val)

            # save the threshold of this iteration
            self.thresholds_array.append(thresh_val)
            # find the confusion matrix for this threshold computation, usingthe groundtruth Y labels vs. the predicted
            # labels
            # predicted lables are in y_prediction
            conf_matrix = confusion_matrix(self.y_test, self.y_prediction)
            fp = conf_matrix[0, 1]
            fpr = fp / self.total_negatives
            self.fpr_array.append(fpr)
            tp = conf_matrix[1, 1]
            tpr = tp / self.total_positives
            self.tpr_array.append(tpr)

        if LOCAL:
            th_str = ""
            f_str = ""
            t_str = ""
            threshold_path = os.path.join(TEMP_DIR, 'threshold.txt')
            fpr_path = os.path.join(TEMP_DIR, 'fpr.txt')
            tpr_path = os.path.join(TEMP_DIR, 'tpr.txt')

            for th, f, t in zip(self.thresholds_array, self.fpr_array, self.tpr_array):
                th_str += str(th) + "\n"
                f_str += str(f) + "\n"
                t_str += str(t) + "\n"

            save_file(content=th_str, filename=threshold_path, method='w')
            save_file(content=f_str, filename=fpr_path, method='w')
            save_file(content=t_str, filename=tpr_path, method='w')

        self.plot_roc_curve(f_array=self.fpr_array, t_array=self.tpr_array)


def calculate_optimum_threshold(f_array, t_array, thresh_array):

    opt_index = 0
    diff = abs(f_array[0] - OPT_THRESH)
    for i, f_value in enumerate(f_array):
        if f_value > OPT_THRESH:
            continue
        if abs(f_value - OPT_THRESH) < diff:
            opt_index = i
            diff = abs(f_value - OPT_THRESH)

    opt_fpr = f_array[opt_index]
    opt_thresh = thresh_array[opt_index]
    opt_tpr = t_array[opt_index]

    print("Optimum Threshold for test data: ", opt_thresh)
    print("test_data FPR: ", opt_fpr)
    print("test_data TPR: ", opt_tpr)

    save_file(content=opt_thresh, filename=OPT_THRESH_PATH, method='w')

    return opt_thresh, opt_fpr, opt_tpr


def mse(image_a, image_b):
    error = np.sum((image_a.astype("float") - image_b.astype("float")) ** 2)
    error /= float(image_a.shape[0] * image_a.shape[1])

    return error


if __name__ == '__main__':
    pass
