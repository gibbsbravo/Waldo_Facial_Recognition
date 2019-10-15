import os
import numpy as np
import pandas as pd
import cv2

import Preprocessing as pp

import keras
import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)

import pytesseract
pytesseract.pytesseract.tesseract_cmd = r"SET PATH TO TESSERACT 3.05 HERE"
#Set path similar to: r"C:\Users\AGB\AppData\Local\Tesseract-OCR\tesseract.exe"

img_input_size = 90
img_input_shape = (img_input_size, img_input_size, 3)
seed=34 
img_classes = np.loadtxt('Trained_Models/img_classes.txt', dtype='str')


def detectNum(inputPath, detectMultiple=False, debug = False):
    """OCR function which can detect numbers from both pictures and videos"""
    if 'jpeg' in inputPath.lower() or 'jpg' in inputPath.lower():
        return pp.detectNum(inputPath, detectMultiple, debug, video_input=False)
    
    elif '.mov' in inputPath.lower()  or '.mp4' in inputPath.lower():
        from collections import Counter
        n_frames = 7
        detected_num_list = []
        
        vidcap = cv2.VideoCapture(inputPath)
        success, img = vidcap.read()
        count = 0
        while success:
            if count % n_frames == 0:
                detected_num_list.append(pp.detectNum(img, detectMultiple, debug, video_input=True))
            else:
                pass
            success,img = vidcap.read()
            count += 1
        
        if detectMultiple:
            flattened_list = []
            for x in detected_num_list:
                for y in x:
                    flattened_list.append(str(y))
                    
            return [a for (a,b) in Counter(flattened_list).most_common() if b > 1]
        else:
            return Counter(detected_num_list).most_common(1)[0][0]
    
    else:
        print('Please select a valid file type: .jpeg, .jpg, .mov, or .mp4')

# Input either the image or video file, set detectMultiple to True for multiple numbers
#detectNum('Individual_Images/Camera1/IMG_3493.jpeg')