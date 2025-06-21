import datetime
import logging
import os
import os.path as osp
import sys
from collections import defaultdict
from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import dlib
import numpy as np
import torch
import torch.optim as optim
import yaml
from imutils import face_utils
from PIL import Image
from skimage import transform as trans
from sklearn import metrics
from torch.nn import DataParallel
from torchvision import transforms as T



class AlignFacePreprocess(object):
    def __init__(self, config: Union[str, dict]):
        self.config = config
        self.face_detector = dlib.get_frontal_face_detector()

        predictor_path = osp.join('preprocess/shape_predictor_81_face_landmarks.dat')
        self.face_predictor = dlib.shape_predictor(predictor_path)
        self.build()
    
    def build(self):
        return 
    
    def __call__(self, image:np.ndarray, inference:bool=False):
        cropped_face, landmarks, _ = self.extract_aligned_face_dlib(image, mask=None)
        return cropped_face
    
    def extract_aligned_face_dlib(self, image:np.ndarray, res:int=256, mask:np.ndarray=None):
        def img_align_crop(img, landmark=None, outsize=None, scale=1.3, mask=None):
            """ 
            align and crop the face according to the given bbox and landmarks
            landmark: 5 key points
            """

            M = None
            target_size = [112, 112]
            dst = np.array([
                [30.2946, 51.6963],
                [65.5318, 51.5014],
                [48.0252, 71.7366],
                [33.5493, 92.3655],
                [62.7299, 92.2041]], dtype=np.float32)

            if target_size[1] == 112:
                dst[:, 0] += 8.0
            
            dst[:, 0] = dst[:, 0] * outsize[0] / target_size[0]
            dst[:, 1] = dst[:, 1] * outsize[1] / target_size[1]

            target_size = outsize

            margin_rate = scale - 1                         #周围有0.15的空白区域
            x_margin = target_size[0] * margin_rate / 2.
            y_margin = target_size[1] * margin_rate / 2.

            # move
            dst[:, 0] += x_margin
            dst[:, 1] += y_margin

            # resize
            dst[:, 0] *= target_size[0] / (target_size[0] + 2 * x_margin)
            dst[:, 1] *= target_size[1] / (target_size[1] + 2 * y_margin)

            src = landmark.astype(np.float32)

            # use skimage tranformation
            tform = trans.SimilarityTransform()
            tform.estimate(src, dst)
            M = tform.params[0:2, :]

            # M: use opencv
            # M = cv2.getAffineTransform(src[[0,1,2],:],dst[[0,1,2],:])

            img = cv2.warpAffine(img, M, (target_size[1], target_size[0]))

            if outsize is not None:
                img = cv2.resize(img, (outsize[1], outsize[0]))
            
            if mask is not None:
                mask = cv2.warpAffine(mask, M, (target_size[1], target_size[0]))
                mask = cv2.resize(mask, (outsize[1], outsize[0]))
                return img, mask
            else:
                return img, None
        def get_keypts(image, face, predictor, face_detector):
            # detect the facial landmarks for the selected face
            shape = predictor(image, face)
            
            # select the key points for the eyes, nose, and mouth
            leye = np.array([shape.part(37).x, shape.part(37).y]).reshape(-1, 2)
            reye = np.array([shape.part(44).x, shape.part(44).y]).reshape(-1, 2)
            nose = np.array([shape.part(30).x, shape.part(30).y]).reshape(-1, 2)
            lmouth = np.array([shape.part(49).x, shape.part(49).y]).reshape(-1, 2)
            rmouth = np.array([shape.part(55).x, shape.part(55).y]).reshape(-1, 2)
            
            pts = np.concatenate([leye, reye, nose, lmouth, rmouth], axis=0)

            return pts
        # Image size
        height, width = image.shape[:2]

        # Convert to rgb
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Detect with dlib
        faces = self.face_detector(rgb, 1)
        if len(faces):
            # For now only take the biggest face
            face = max(faces, key=lambda rect: rect.width() * rect.height())
            
            # Get the landmarks/parts for the face in box d only with the five key points
            landmarks = get_keypts(rgb, face, self.face_predictor, self.face_detector)

            # Align and crop the face
            cropped_face, mask_face = img_align_crop(rgb, landmarks, outsize=(res, res), mask=mask)
            cropped_face = cv2.cvtColor(cropped_face, cv2.COLOR_RGB2BGR)
            
            # Extract the all landmarks from the aligned face
            face_align = self.face_detector(cropped_face, 1)  
            if len(face_align) == 0:    #变换完之后再check一遍脸到底是否还存在
                return None, None, None
            landmark = self.face_predictor(cropped_face, face_align[0])
            landmark = face_utils.shape_to_np(landmark)

            return cropped_face, landmark, mask_face
        
        else:
            return None, None, None



class ToTensor(object):
    def __init__(self, config: Union[str, dict]):
        self.config = config
        self.build()
    
    def load(self, img: Union[np.ndarray]) -> Union[Image.Image, None]:

        if img is None:
            return None
        
        size = self.config["resolution"]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (size, size), interpolation=cv2.INTER_CUBIC)
        return Image.fromarray(np.array(img, dtype=np.uint8))
    
    
    def build(self):
        mean = self.config['mean']
        std = self.config['std']

        self.transform = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=mean, std=std)
        ])
    

    def __call__(self, image:np.ndarray, inference:bool=False) -> dict:
        image = self.load(image)
        image_tensor = self.transform(image)
        return image_tensor




    