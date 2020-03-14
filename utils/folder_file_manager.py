import os
import ntpath
import logging


def make_directory_if_not_exists(dir_path):

    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    return dir_path


def get_index_from_file_path(path):

    try:
        file_parent_path, file_name = ntpath.split(path)
        base_name, extension = os.path.splitext(file_name)
        index = int(base_name[base_name.rfind("_") + 1:])
    except Exception as e:
        file_name = ""
        index = ""
        print(e)

    return file_name, index


def load_text(filename):

    if os.path.isfile(filename):
        file = open(filename, 'r')
        text = file.read()
        file.close()
    else:
        text = ''

    return text


def save_file(content, filename, method):

    file = open(filename, method)
    file.write(content)
    file.close()

    return


def log_print(info_str, only_print=False):

    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(message)s',
                        datefmt='%d/%b/%Y %H:%M:%S',
                        filename='result.log'
                        )
    # print(info_str)

    if not only_print:
        logging.info(info_str)


def extract_file_name(file_path):

    file_name_ext = ntpath.basename(file_path)
    file_name = file_name_ext.replace(file_name_ext[file_name_ext.rfind("."):], "")

    return file_name
