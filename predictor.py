"""
eval pretained model.
"""
import argparse
import datetime
import json
import os

import os.path as osp
import pickle
import random
import time
import traceback
from collections import defaultdict
from copy import deepcopy
from os.path import join

import cv2
import numpy as np
import pandas as pd
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import yaml
from PIL import Image as pil_image
from tqdm import tqdm

from dataset.abstract_dataset import DeepfakeAbstractBaseDataset
from detectors.xception_detector import XceptionDetector
from preprocess.preprocessor import AlignFacePreprocess, ToTensor


def parse_args():
    parser = argparse.ArgumentParser(description='predicter')
    parser.add_argument('--input_csv', type=str, default="example/example_input1.csv")
    parser.add_argument("--output_csv", type=str, default="example/output.csv")
    args = parser.parse_args()
    return args




class BasePredictor(object):
    def __init__(self, name=None):
        if name is not None:
            self.module_name = name
    
    def preprocess_one_image(self, image:np.ndarray):
        pass
    
    def predict_one_image(self, image_path:str):
        pass
    
    



class BaselinePredictor(BasePredictor):
    def __init__(self,):
        super().__init__("BaselinePredictor")
        self.model, self.preprocessor, self.face_extractor = self.build_model()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    
    def init_seed(self,config):
        if config['manualSeed'] is None:
            config['manualSeed'] = random.randint(1, 10000)
        random.seed(config['manualSeed'])
        torch.manual_seed(config['manualSeed'])
        if config['cuda']:
            torch.cuda.manual_seed_all(config['manualSeed'])


    def build_model(self,):
        config = yaml.safe_load(open("configs/xception.yaml"))
        config2 = yaml.safe_load(open("configs/test_config.yaml"))
        config.update(config2)
        cudnn.benchmark = True

        model = XceptionDetector(config)

        self.init_seed(config)
        ckpt = torch.load("./exp/ckpt_best.pth")
        model.load_state_dict(ckpt)
        model.eval()

        face_extractor = AlignFacePreprocess(config)
        preprocess = ToTensor(config)
        return model, preprocess, face_extractor
    
    def preprocess_one_image(self, image_path:str):
        image = cv2.imread(image_path)
        croped_face = self.face_extractor(image)
        input_tensor = self.preprocessor(croped_face).unsqueeze(0).to(self.device)

        return input_tensor


    def predict_one_image(self, image_path:str):
        try:
            input_tensor = self.preprocess_one_image(image_path)
            score = self.model({"image": input_tensor})["prob"].detach().item()
        except:
            print(image_path)
            score = -1
        return score



class BatchBaselineDataset(torch.utils.data.Dataset):
    def __init__(self, image_paths:str, workers:int=8):
        self.image_paths = image_paths
        self.workers = workers
        self.preprocessors = []
        self.face_extractors = []
        for _ in range(self.workers):
            preprocessor, face_extractor = self.build_model()
            self.preprocessors.append(preprocessor)
            self.face_extractors.append(face_extractor)

    def __len__(self):
        return len(self.image_paths)
    
    def build_model(self,):
        config = yaml.safe_load(open("configs/xception.yaml"))
        config2 = yaml.safe_load(open("configs/test_config.yaml"))
        config.update(config2)

        face_extractor = AlignFacePreprocess(config)
        preprocess = ToTensor(config)
        return preprocess, face_extractor

    def preprocess_one_image(self, image_path:str, index:int):
        image = cv2.imread(image_path)
        croped_face = self.face_extractors[index % self.workers](image)
        if croped_face is None:
            croped_face = cv2.resize(image, (256, 256))
        input_tensor = self.preprocessors[index % self.workers](croped_face)
        return input_tensor

    def __getitem__(self, index):
        return self.preprocess_one_image(self.image_paths[index], index), self.image_paths[index]



class BatchBaselinePredictor(BasePredictor):
    def __init__(self,):
        super().__init__("BaselinePredictor")
        self.model = self.build_model()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    
    def init_seed(self,config):
        if config['manualSeed'] is None:
            config['manualSeed'] = random.randint(1, 10000)
        random.seed(config['manualSeed'])
        torch.manual_seed(config['manualSeed'])
        if config['cuda']:
            torch.cuda.manual_seed_all(config['manualSeed'])


    def build_model(self,):
        config = yaml.safe_load(open("configs/xception.yaml"))
        config2 = yaml.safe_load(open("configs/test_config.yaml"))
        config.update(config2)
        cudnn.benchmark = True

        model = XceptionDetector(config)

        self.init_seed(config)
        ckpt = torch.load("./ckpt_best.pth")
        model.load_state_dict(ckpt)
        model.eval()

        return model
    

    def predict(self, input_tensor:torch.Tensor):
        try:
            score = self.model({"image": input_tensor})["prob"].detach().cpu().numpy().tolist()
        except:
            traceback.print_exc() 
            score = [-1] * input_tensor.shape[0]
        return score

  

def predict_single():

    args = parse_args()    
    predictor = BaselinePredictor()
    # score = predictor.predict_one_image("./example/example1.jpg")
    input_csv = args.input_csv
    output_csv = args.output_csv

    
    res = []
    df = pd.read_csv(input_csv)
    for index, row in tqdm(df.iterrows()):
        image_path = row['image_name']
        score = predictor.predict_one_image(image_path)
        res.append([osp.basename(image_path), score])
    
    
            
    df_result = pd.DataFrame(res, columns=["image_name", "score"])
    df_result.to_csv(output_csv, index=False)
    



def predict_batch():
    args = parse_args()
    predictor = BatchBaselinePredictor()

    input_csv = args.input_csv
    output_csv = args.output_csv

    image_paths = [] 
    df = pd.read_csv(input_csv)
    # for _ in range(10000):
    for index, row in df.iterrows():
        image_paths.append(row['image_name'])
    
    dataset = BatchBaselineDataset(image_paths)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=False, num_workers=8)

    res = []
    for images, image_paths in tqdm(dataloader):
        images = images.to(predictor.device)
        scores = predictor.predict(images)
        for image_path, score in zip(image_paths, scores):
            res.append([osp.basename(image_path), score])

    df_result = pd.DataFrame(res, columns=["image_name", "score"])
    df_result.to_csv(output_csv, index=False)
    



if __name__ == '__main__':
    # predict_single()
    predict_batch()
