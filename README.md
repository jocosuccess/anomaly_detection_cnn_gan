# CNN_GAN_Cascade_Anomaly_Detection_A_V

## Overview

This project is to detect the anomaly of images using CNN and GAN. In the first step of this project, the fpr and 
tpr values according to all the threshold values are obtained while running projects with 2000 normal images(negative) 
and 102 abnormal images(positive) and then the ROC curve is plotted. Then the optimum threshold value can be obtained by

## Structure

- src
    
    The source code to make the model and estimate the abnormal images with the model.

- utils

    * The weights and model for this project
    * The source code to manage the folder and files in the project

- app

    The main execution file
    
- requirements
    
    * All the dependencies for this project
    * When using GPU with CUDA 9.0, the version of some dependencies are as following:
    ```
        tensorflow-gpu==1.12.0
        keras==0.23.2
        matplotlib==2.0.0
        pandas==0.9
        numpy==1.18.1
    ``` 

- test_data

    This directory contains all the test data images, one of which named positives contains 102 abnormal images and 
    other of which named HealthyAll contains more than 80000 normal images. But since it takes more than a week for all 
    the test images of only 10 threshold values(2 ~ 11) even if using GPU(970), when obtaining the values of fpr and tpr
    of 10 threshold values, only 2000 normal images and 102 abnormal images are used instead of 80000 normal images. But 
    when estimating the images for the optimum threshold value, all the images(80000, 102) are used.
    
    Therefore, the directory to contain 2000 normal images is needed.

- sub_healthy

    The directory to contain 2000 normal images necessary for getting the fpr and tpr value of all the threshold values.

- settings

    * The  path and name of several directories
    * The  path and name of the model and weights for this project 
    * The path and name of result files(csv, png)
    * The value of max gan loss and min gan loss, and the base value necessary for finding optimum threshold value

## Installation

- Environment

    Ubuntu 16.04+, Python 3.5+, 

- Dependency Installation

    ```
        pip3 intall -r requirements.txt
    ```
  
    * Note: if using GPU, please refer the requirements section in Structure.

## Execution

- Please copy the test data images into positive and HealthyAll. If having to use sub_healthy directory, please copy the
necessary amount of normal images.

- Please go ahead the directory in this project, and run the following command in terminal.

    ```
        python3 app.py
    ```

- While running project, the result values(threshold, fpr, tpr) are saved in result/processed_thresh_fpr_tpr.csv file 
every time for each threshold value. Though the project is stopped by a specific reason, when running it again, it will
start from the last saved threshold value to save processing time. 

Finally, after obtaining all the values(fpr, tpr) for all the threshold values, the optimum threshold value will be 
extracted, which is saved in result/opt_threshold.txt. After that, the process of estimating all the test images with 
the optimum threshold value continues to be performed. The result of the final process is saved in result/result.csv.
