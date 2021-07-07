import os

import numpy as np

TEST_TIME_STEPS = [1.0, 2.0, 3.0, 4.0, 5.0]
TEST_RAW_OUTPUT = np.array(
    [
        [1.0, 1.0, 1.0, 1.0, 1.0],
        [2.0, 2.0, 2.0, 2.0, 2.0],
        [3.0, 3.0, 3.0, 3.0, 3.0],
        [4.0, 4.0, 4.0, 4.0, 4.0],
    ]
)
dirpath = os.path.dirname(os.path.abspath(__file__))
TEST_RAW_CSV_FILE_PATH = os.path.join(dirpath, "./test_raw_file.csv")
TEST_MAGNET_SPRING_FEA_PATH = os.path.join(dirpath, "./test_magnetic_spring_fea.csv")
TEST_ACCELEROMETER_FILE_PATH = os.path.join(dirpath, "./test_accelerometer_file.csv")
