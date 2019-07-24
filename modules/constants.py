import os
import time


# main directory path
PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


RUN_NAME = "RUN_"+time.strftime("%Y%m%d_%H%M%S")


# path to individual directories
DATA_DIR = os.path.join(PATH, "data/")
RUN_ARCHIVE_DIR = os.path.join(DATA_DIR, RUN_NAME + "/"); os.mkdir(RUN_ARCHIVE_DIR)
DIRS = [DATA_DIR, RUN_ARCHIVE_DIR]

# check directories for their existence
for a_dir in DIRS:
    if os.path.isdir(a_dir) is False:
        raise FileNotFoundError("The directory "+a_dir+" was not found. Please set it up before running.")
