import logging
import onnxruntime
import torch
import cv2
import numpy as np
from PIL import Image
import os
import io
import urllib.request
from ts.torch_handler.object_detector import ObjectDetector
from util_v8 import *
import base64
import json
from typing import List, Dict, Any, Tuple
from Antispoof import AntiSpoof




class FaceDetection(ObjectDetector):
    def __init__(self):
        self.session = None
        self.providers = ['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider']
        self.img = []
        self.blob = None
        self.classes = ['human']
        self.antisp = AntiSpoof()

    def initialize(self, context):
        self._context = context
        self.manifest = context.manifest
        properties = context.system_properties
        model_dir = properties.get("model_dir")
        serialized_file = self.manifest['model']['serializedFile']
        model_file_path = os.path.join(model_dir, serialized_file)
        sess_options = onnxruntime.SessionOptions()
        self.session = onnxruntime.InferenceSession(model_file_path, sess_options=sess_options, providers=['CPUExecutionProvider'])

    @staticmethod
    def extract_inner_dicts(dictionary):
        dicts_list = []
        for value in dictionary.values():
            if isinstance(value, dict):
                dicts_list.extend(FaceDetection.extract_inner_dicts(value))
        if not any(isinstance(value, dict) for value in dictionary.values()):
            dicts_list.append(dictionary)
        return dicts_list
    
    def preprocess(self, data):
        images = []
        self.img_size=[]
        for data_get in data:
            data_get = data_get.get("data") or data_get.get("body")
            imgs = FaceDetection.extract_inner_dicts(data_get)            
            # print('Cam_id: ', imgs[0]['id'])
            
            for thing in imgs:
                image = thing['image']
                if isinstance(image, str):
                    # if the image is a string of bytesarray.
                    # req = urllib.request.urlopen(data_get)
                    image = base64.b64decode(image)
                    image = Image.open(io.BytesIO(image))

                # If the image is sent as bytesarray
                else:
                    # if the image is a list  
                    byte_data = io.BytesIO(image)
                    image = Image.open(byte_data)

                preprocess_img = np.array(image).astype(np.uint8)
                self.img.append(preprocess_img)
                                
                image = np.expand_dims(preprocess_img, axis=0).astype('float32') / 255.
                image = np.transpose(image, [0, 3, 1, 2])
                self.img_size.append(thing['size'])
                # print(image.shape, thing['size'])
                images.append({'id': thing['id'],
                               'frame': thing['frame'],
                            'image': image})

        return images
        


    def inference(self, blob):
        self.blob = blob
        batch_image = np.array([img['image'].squeeze() for img in blob])
        ids = [img['id'] for img in blob]
        frames = [img['frame'] for img in blob]
        results = self.session.run(None, {self.session.get_inputs()[0].name: batch_image})[0]
        return ids, frames, results

    def postprocess(self, preds):
        ids = preds[0]
        frames = preds[1]
        preds = preds[2]

        # pred = pred.reshape(len(ids), 20, 8400)
        res = []        
        dim=3

        preds = non_max_suppression(torch.tensor(preds), conf_thres=0.7)

        # print(self.img_size, self.img[0].shape)
        result = postprocess(preds, self.img[0], self.img_size, dim)
        # print(result)
        bboxs = result['boxes']
        # keypoints = result['keypoints']
        classes = result['classes']
        scores = result['scores']
        is_real = []
        
        for i, f, b, c, s, image, orig_img_shape in zip(ids, frames, bboxs, classes, scores, self.img, self.img_size):
            b = b.squeeze().to(torch.int).tolist()
            cl = c.squeeze().tolist()
            s = s.squeeze().tolist()
            # k = k[:, :, :-1].squeeze().tolist()
            # print(k)
            # cv2.imshow('orig', image)
            if len(b) != 0 and isinstance(b[0], int):
                b = [b]
            # Scale back the coordinates
            for box in b:
                x, y, w, h = box
                img = cv2.resize(image, (orig_img_shape[1], orig_img_shape[0]), interpolation = cv2.INTER_LINEAR)
                # x += int(orig_img_shape[0]/image.shape[0])
                # y += int(orig_img_shape[1]/image.shape[1])
                # w += int(orig_img_shape[0]/image.shape[0])
                # h += int(orig_img_shape[1]/image.shape[1])
                
                ir,_ = self.antisp.analyze(img, (x, y, w, h))
                is_real.append(ir)
                
                
            # for box in b:
            #     is_real, antiscore = self.antisp.analyze(image, box)
            
            if isinstance(cl, float):
                cl = [cl]
            res.append({
                "id": i,
                "frame": f,
                "bbox": b,
                "class": [self.classes[int(c)] for c in cl],
                "score": s,
                "is_real": is_real,})
            # print(res)
            
        self.img = []
        return [res]
        
    
    

    
    




    
    
    