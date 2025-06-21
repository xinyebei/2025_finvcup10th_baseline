'''
description: 
param : 
return: 
Author: xinyebei@xinye.com
Date: 2025-04-23 14:37:05
LastEditors: xinyebei@xinye.com
'''
import argparse
import json
import os
import os.path as osp


def parse_args():
    parser = argparse.ArgumentParser(description='Prepare dataset info')
    parser.add_argument('--dataset_root_path', type=str, default='./data/train', help='dataset root path')
    parser.add_argument('--label_info', type=str, default='./data/train.txt', help='label info')
    parser.add_argument('--mode', type=str, default='train', help='build dataset mode')
    parser.add_argument('--dataset_name', type=str, default='train', help='dataset name')

    parser.add_argument('--dataset_info_path', type=str, default='./data/dataset_info', help='dataset info path')

    args = parser.parse_args()
    return args


if __name__ == '__main__':

    args = parse_args()
    dataset_dict = {}
    dataset_dict = {"real":{"frames":[],}, "fake":{"frames":[],}}

    with open(args.label_info, 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            if args.mode == "train" or args.mode == "val" or args.mode == "test":
                name, label = line.split(" ")
                
            label = int(label)
            name = osp.join(args.dataset_root_path, name)
            if osp.exists(name):
                if label == 0:
                    dataset_dict["real"]["frames"].append(name)
                else:
                    dataset_dict["fake"]["frames"].append(name)
        



    # Convert the dataset dictionary to JSON format and save to file
    output_file_path = os.path.join(args.dataset_info_path, f"{args.dataset_name}.json")
    if not osp.exists(args.dataset_info_path):
        os.makedirs(args.dataset_info_path, exist_ok=True)
    with open(output_file_path, 'w') as f:
        json.dump(dataset_dict, f)
    # print the successfully generated dataset dictionary
    print(f"{args.dataset_name}.json generated successfully.")