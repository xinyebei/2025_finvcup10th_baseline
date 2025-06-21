import argparse
import concurrent.futures
import datetime
import glob
import logging
import os
import os.path as osp
import sys
import time
import traceback
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import cv2
import dlib
import numpy as np
import yaml
from imutils import face_utils
from skimage import transform as trans
from tqdm import tqdm

from preprocess.preprocessor import AlignFacePreprocess


def parse_args():
    parser = argparse.ArgumentParser(description="Preprocess images")
    parser.add_argument("--image_path", type=str, required=False, help="Path to the image", default="")
    parser.add_argument("--mask_path", type=str, required=False, help="Path to the mask image", default=None)
    parser.add_argument("--save_path", type=str, required=False, help="Path to the save root", default="")
    parser.add_argument("--total_splits", type=int, required=False, help="total splits in multiprocessing", default=4)
    parser.add_argument("--splits", type=int, required=False, help="selected splits", default=0)

    args = parser.parse_args()
    return args



if __name__ == "__main__":
    args = parse_args()
    image_path = args.image_path
    mask_path = args.mask_path

    
    image_files = os.listdir(image_path)
    image_files = [osp.join(image_path, f) for f in image_files if f.endswith('.jpg') or f.endswith('.png') or f.endswith('.jpeg')]
    image_files = sorted(image_files)
    image_files = [Path(i) for i in image_files]

    
    mask_files = [] 
    if mask_path is not None:
        mask_files = os.listdir(mask_path)
        assert len(image_files) == len(mask_files), "The number of images and masks should be the same!!!"

        mask_files = [osp.join(mask_path, f) for f in mask_files if f.endswith('.jpg') or f.endswith('.png')]
        mask_files = sorted(mask_files)
        mask_files = [Path(i) for i in mask_files]
    
    preprocessor = AlignFacePreprocess(None)
    
    save_crop_root = osp.join(args.save_path, "crop")

    if not osp.exists(save_crop_root):
        os.makedirs(save_crop_root, exist_ok=True)
    

    def task_func(image_file, mask_path, save_crop_name,):
        image = cv2.imread(image_file) 
        cropped_face = preprocessor(image)

        if cropped_face is not None:
            cv2.imwrite(save_crop_name, cropped_face)
        else:
            print(f"[warning!!!!] preprocess failed for {image_file}, using origin image instead")
            cv2.imwrite(save_crop_name, image)
    
        
    params_list = []
    for image_file in image_files:
        image_name = image_file.name
        image_stem = image_file.stem
        save_crop_name = osp.join(save_crop_root, image_name)
        params_list.append((image_file, mask_path, save_crop_name))

    
    
    splits = np.array_split(params_list, args.total_splits)
    params_list = splits[args.splits]
    
    
    with tqdm(total=len(params_list), desc="Processing") as pbar:
        for params in params_list:
            task_func(*params)
            pbar.update()

    
    
    
    

