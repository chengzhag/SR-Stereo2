import os
import os.path
import math
import random
from utils import myUtils

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
    '.pfm', 'PFM'
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def _scanImages(filepath, episodes, folderName):
    folderDir = os.path.join(filepath, episodes, folderName)
    imagedirs = [os.path.join(folderDir, d) for d in os.listdir(folderDir) if is_image_file(d)]
    imagedirs.sort()
    return imagedirs


def dataloader(filepath, trainProportion=0.8, shuffle=False):
    episodes = [d for d in os.listdir(filepath) if os.path.isdir(os.path.join(filepath, d))]
    episodes.sort()

    all_left_img = []
    all_right_img = []
    all_left_disp = []
    all_right_disp = []
    test_left_img = []
    test_right_img = []
    test_left_disp = []
    test_right_disp = []

    for i, episode in enumerate(episodes):
        if i + 1 <= math.ceil(trainProportion * len(episodes)):
            all_left_img += (_scanImages(filepath, episode, 'Camera2RGB'))
            all_right_img += (_scanImages(filepath, episode, 'Camera3RGB'))
            all_left_disp += (_scanImages(filepath, episode, 'Camera2Depth'))
            all_right_disp += (_scanImages(filepath, episode, 'Camera3Depth'))
        else:
            test_left_img += (_scanImages(filepath, episode, 'Camera2RGB'))
            test_right_img += (_scanImages(filepath, episode, 'Camera3RGB'))
            test_left_disp += (_scanImages(filepath, episode, 'Camera2Depth'))
            test_right_disp += (_scanImages(filepath, episode, 'Camera3Depth'))

    return all_left_img, all_right_img, all_left_disp, all_right_disp, test_left_img, test_right_img, test_left_disp, test_right_disp

