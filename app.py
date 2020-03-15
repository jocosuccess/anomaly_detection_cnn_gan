from src.whole_pipeline_new import WholePipeline
from settings import TEST_DATA_DIR, OPT_THRESH_PATH
from utils.folder_file_manager import load_text


if __name__ == '__main__':

    whole_pipeline = WholePipeline()
    opt_thresh_value = load_text(filename=OPT_THRESH_PATH)

    whole_pipeline.collect_fpr_tpr_all_thresh()
    if opt_thresh_value != "":
        whole_pipeline.get_fpr_tpr(dir_path=TEST_DATA_DIR, thresh=float(opt_thresh_value), opt_thresh=True)
    else:
        print("There is no optimum thresh value")
