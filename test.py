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
from logger import create_logger
from metrics.base_metrics_class import Recorder
from metrics.utils import get_test_metrics
from trainer import Trainer

parser = argparse.ArgumentParser(description='Process some paths.')
parser.add_argument('--detector_path', type=str, 
                    default="./configs/xception.yaml",
                    help='path to detector YAML file')
parser.add_argument("--test_dataset", nargs="+", default=["test1_infer"])
parser.add_argument("--mode", type=str, default="inference", choices=["eval_metric", "inference"])
parser.add_argument("--save_res_path", type=str, default="./submib/test1.csv",)

parser.add_argument('--weights_path', type=str, 
                    default="",
                    )
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def init_seed(config):
    if config['manualSeed'] is None:
        config['manualSeed'] = random.randint(1, 10000)
    random.seed(config['manualSeed'])
    torch.manual_seed(config['manualSeed'])
    if config['cuda']:
        torch.cuda.manual_seed_all(config['manualSeed'])


def prepare_testing_data(config):
    def get_test_data_loader(config, test_name):
        # update the config dictionary with the specific testing dataset
        config = config.copy()  # create a copy of config to avoid altering the original one
        config['test_dataset'] = test_name  # specify the current test dataset
        test_set = DeepfakeAbstractBaseDataset(
                config=config,
                # mode='test', 
                mode=config["mode"], 
            )
        print(test_name)
        print(len(test_set))
        test_data_loader = \
            torch.utils.data.DataLoader(
                dataset=test_set, 
                batch_size=config['test_batchSize'],
                shuffle=False, 
                num_workers=int(config['workers']),
                collate_fn=test_set.collate_fn,
                drop_last=False
            )
        return test_data_loader

    test_data_loaders = {}
    for one_test_name in config['test_dataset']:
        test_data_loaders[one_test_name] = get_test_data_loader(config, one_test_name)
    return test_data_loaders


def choose_metric(config):
    metric_scoring = config['metric_scoring']
    if metric_scoring not in ['eer', 'auc', 'acc', 'ap']:
        raise NotImplementedError('metric {} is not implemented'.format(metric_scoring))
    return metric_scoring


def test_one_dataset(model, data_loader):
    prediction_lists = []
    feature_lists = []
    label_lists = []
    image_paths = []
    for i, data_dict in tqdm(enumerate(data_loader), total=len(data_loader)):
        # get data
        data, label, mask, landmark, image_path = \
        data_dict['image'], data_dict['label'], data_dict['mask'], data_dict['landmark'], data_dict["image_paths"][0]
        label = torch.where(data_dict['label'] != 0, 1, 0)
        # move data to GPU
        data_dict['image'], data_dict['label'] = data.to(device), label.to(device)
        if mask is not None:
            data_dict['mask'] = mask.to(device)
        if landmark is not None:
            data_dict['landmark'] = landmark.to(device)

        # model forward without considering gradient computation
        predictions = inference(model, data_dict)
        label_lists += list(data_dict['label'].cpu().detach().numpy())
        prediction_lists += list(predictions['prob'].cpu().detach().numpy())
        feature_lists += list(predictions['feat'].cpu().detach().numpy())
        image_paths.append(osp.join(image_path))
    
    return np.array(prediction_lists), np.array(label_lists), image_paths
    
def test_epoch(model, test_data_loaders, threshold=0.5, mode="train"):
    # set model to eval mode
    model.eval()

    # define test recorder
    metrics_all_datasets = {}

    # testing for all test data
    keys = test_data_loaders.keys()
    for key in keys:
        data_dict = test_data_loaders[key].dataset.data_dict
        # compute loss for each dataset

        predictions_nps, label_nps, image_names = test_one_dataset(model, test_data_loaders[key])
        if mode == "train" or mode == "test":
            # compute metric for each dataset
            metric_one_dataset = get_test_metrics(y_pred=predictions_nps, y_true=label_nps,
                                                img_names=data_dict['image'], 
                                                threshold_acc=threshold,
                                                )
            metric_one_dataset["image_names"] = image_names
            metrics_all_datasets[key] = metric_one_dataset
            
            
            # info for each dataset
            tqdm.write(f"dataset: {key}")
            for k, v in metric_one_dataset.items():
                if k in ["acc","auc","err","ap",]:
                    tqdm.write(f"{k}: {v}")
                else:
                    for name in ["recall@0.1", "recall@0.3","recall@0.5","recall@0.7", "recall@0.9", 
                                "precision@0.1", "precision@0.3", "precision@0.5", "precision@0.7", "precision@0.9",
                                "f1@0.1", "f1@0.3","f1@0.5","f1@0.7","f1@0.9",
                                ]:
                        if 'fake' in k and name in k:
                            tqdm.write(f"{k}: {v}")

        elif mode == "inference":
            metric_one_dataset = {}
            metric_one_dataset ["image_names"] = image_names
            metric_one_dataset ["scores"] = predictions_nps.tolist()
            metrics_all_datasets[key] = metric_one_dataset
            

    return metrics_all_datasets

@torch.no_grad()
def inference(model, data_dict):
    predictions = model(data_dict, inference=True)
    return predictions


def main():
    # parse options and load config
    with open(args.detector_path, 'r') as f:
        config = yaml.safe_load(f)
    with open('./configs/test_config.yaml', 'r') as f:
        config2 = yaml.safe_load(f)
    config.update(config2)
    if 'label_dict' in config:
        config2['label_dict']=config['label_dict']
    weights_path = None
    # If arguments are provided, they will overwrite the yaml settings
    if args.test_dataset:
        config['test_dataset'] = args.test_dataset
    if args.weights_path:
        config['weights_path'] = args.weights_path
        weights_path = args.weights_path
    

    config["mode"] = "test"
    if args.mode == "inference":
        config["mode"] = "inference"

    config["threshold"] = 0.5
    # init seed
    init_seed(config)

    # set cudnn benchmark if needed
    if config['cudnn']:
        cudnn.benchmark = True

    # prepare the testing data loader
    test_data_loaders = prepare_testing_data(config)
    # return
    # prepare the model (detector)
    model_class = XceptionDetector

    model = model_class(config).to(device)
    epoch = 0
    if weights_path:
        try:
            epoch = int(weights_path.split('/')[-1].split('.')[0].split('_')[2])
        except:
            epoch = 0
        ckpt = torch.load(weights_path, map_location=device)
        new_state_dict = {k.replace('module.', ''): v for k, v in ckpt.items()}
        

        model.load_state_dict(new_state_dict, strict=True)
        print('===> Load checkpoint done!')
    else:
        print('Fail to load the pre-trained weights')
    
    # start testing
    best_metric = test_epoch(model, test_data_loaders, threshold=config['threshold'], mode=config["mode"])
    best_metric["threshold"] = config["threshold"]

    
    
   
    if config["mode"] == "inference":

        results = [] 
        for name, score in zip(best_metric[config['test_dataset'][0]]["image_names"], best_metric[config['test_dataset'][0]]["scores"]):
            results.append([name, score])
            
        df_result = pd.DataFrame(results, columns=["image_names", "scores"])
        df_result.to_csv(args.save_res_path, index=False, header=None)
    
if __name__ == '__main__':
    main()
