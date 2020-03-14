import os

from utils.folder_file_manager import make_directory_if_not_exists

CUR_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGE_RESULT_DIR = make_directory_if_not_exists(os.path.join(CUR_DIR, 'result'))
WEIGHTS_DIR = make_directory_if_not_exists(os.path.join(CUR_DIR, 'utils', 'weights'))
TEST_DATA_DIR = os.path.join(CUR_DIR, 'test_data')
POSITIVE_DIR = os.path.join(TEST_DATA_DIR, 'positives')
NEGATIVE_DIR = os.path.join(TEST_DATA_DIR, 'HealthyAll')
HEALTHY_DIR = "/media/mensa/Entertainment/DemoTest/HealthyAll"

ROC_CURVE_PATH = os.path.join(IMAGE_RESULT_DIR, 'roc_curve.png')
MODEL_PATH = os.path.join(CUR_DIR, 'utils', 'NoisyVSTissueModel.h5')
GENERATOR_WEIGHT = os.path.join(WEIGHTS_DIR, 'generator_127.h5')
DISCRIMINATOR_WEIGHT = os.path.join(WEIGHTS_DIR, 'discriminator_127.h5')
OPT_THRESH_PATH = os.path.join(IMAGE_RESULT_DIR, 'opt_threshold.txt')
OPT_RESULT_CSV = os.path.join(IMAGE_RESULT_DIR, 'result.csv')

MAX_GAN_LOSS_MSE = 99
MIN_GAN_LOSS_MSE = 1
OPT_THRESH = 0.0005
LOCAL = True
if LOCAL:
    TEMP_DIR = make_directory_if_not_exists(os.path.join(CUR_DIR, 'temp'))
