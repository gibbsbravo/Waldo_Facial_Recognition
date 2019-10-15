import os
import numpy as np
import pandas as pd
import cv2

import Preprocessing as pp
import Face_Recognition as fr

import keras
import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)

def detect_faces(input_path):
    """Detect faces in the image using Haar filters"""
    img = cv2.imread(input_path)
    gray = pp.grayscale(img);
    
    if img.shape[1] < img.shape[0]:
        face_cascade = cv2.CascadeClassifier('Haar_Filters/haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(gray,1.05,20)
        
        if len(faces) == 0:
            face_cascade = cv2.CascadeClassifier('Haar_Filters/haarcascade_profileface.xml')
            faces = face_cascade.detectMultiScale(gray,1.05,20)
            
            if len(faces) == 0:
                face_cascade = cv2.CascadeClassifier('Haar_Filters/haarcascade_frontalface_default.xml')
                faces = face_cascade.detectMultiScale(gray)
        
    else:
        face_cascade = cv2.CascadeClassifier('Haar_Filters/haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(gray,1.05,4)
    
    image_array = []
    x_coordinate = []
    y_coordinate = []
    opencv_coordinates = []
    
    for idx, (x,y,w,h) in enumerate(faces):
        face = img[y:y+h, x:x+w]
        if len(face_cascade.detectMultiScale(face,1.05,6)) == 1:
            image_array.append(face)
            x_coordinate.append(int(x+w/2))
            y_coordinate.append(int(y+h/2))
            opencv_coordinates.append((x,y,w,h))
    image_array = np.array(image_array)
    x_coordinate, y_coordinate  = np.array(x_coordinate), np.array(y_coordinate)
    opencv_coordinates = np.array(opencv_coordinates)
    
    return image_array, x_coordinate, y_coordinate, opencv_coordinates

def resize_array(image_array, img_input_shape):
    """Resize the input array"""
    image_array = [pp.resize_image(x,img_input_shape[0]) for x in image_array]
    
    # Ensure shape matches exactly 
    for i,img in enumerate(image_array):
        shape_delta = img_input_shape[0] - img.shape[0]
        if shape_delta > 0:
            new_row = np.random.randint(0,255,[shape_delta,img_input_shape[1],img_input_shape[2]],dtype='uint8')
            image_array[i] = np.vstack([image_array[i],new_row])
            
        elif shape_delta < 0:
            image_array[i] = image_array[i][:img_input_shape[0],:,:]
    
    # Ensure type is uint8 for HOG & Surf
    image_array = np.array([x.astype('uint8') for x in image_array])
    return image_array

def extract_features(image_array, featureType):
    """Apply feature extraction approach"""
    if featureType == 'ORB':
        # Load Kmeans model
        kmeans_model = fr.load_model('Trained_Models/Kmeans_model.sav')
        ORB_features = fr.extract_ORB_features(image_array, kmeans_model, normalize=False)
        features = ORB_features.copy()
        
    elif featureType == 'HOG':
        features = fr.extract_HOG_features(image_array)
        
    elif featureType == 'Flatten':
        features = fr.flatten_array(image_array)
        
    else:
        print("Please select valid feature type: 'ORB', 'HOG', or 'Flatten'")
        features = image_array
        
    return features

def predict_emotion(image_array, model_name):
    """Predict the emotion based on the trained model"""
    text_labels = {0:'Angry', 1:'Disgust', 2:'Fear', 3:'Happy', 4:'Sad', 5:'Surprised', 6:'Neutral'}
    model = fr.load_model(model_name)
    image_array_HOG = fr.extract_HOG_features(image_array)
    
    pred_emotion = [model.classes_[i] for i in model.predict_proba(image_array_HOG).argmax(axis=-1)]
    emotion_label = np.vectorize(text_labels.get)(pred_emotion)
    
    return emotion_label

# %% RecogniseFace Function

img_input_size = 90
img_input_shape = (img_input_size, img_input_size, 3)
seed=34
img_classes = np.loadtxt('Trained_Models/img_classes.txt', dtype='str')


def RecogniseFace(inputPath, featureType, classifierName, showImage = False):
    """Main function for facial recognition which returns dataframe of image label predictions"""
    image_array, x_coordinate, y_coordinate, opencv_coordinates = detect_faces(inputPath)
    
    if image_array.size == 0:
        face_matrix = pd.DataFrame([])
        
    else:
        image_array = resize_array(image_array, img_input_shape)
    
        if classifierName == 'CNN':
            # load model
            model = keras.models.load_model('Trained_Models/VGG_model.h5')
            predictions = model.predict(image_array)
            pred_class = np.array([img_classes[i] for i in predictions.argmax(axis=-1)])
        
        else:    
            extracted_features = extract_features(image_array, featureType)
                 
            if classifierName in ['SVM', 'Logistic_Regression', 'Naive_Bayes']:
                model = fr.load_model('Trained_Models/'+classifierName+'_'+featureType+'.sav')
                pred_class = [img_classes[i] for i in model.predict_proba(extracted_features).argmax(axis=-1)]
                
            else:
                print("Please select valid classifier name: 'CNN', 'SVM', 'Logistic_Regression', or 'Naive_Bayes'")
        
        pred_emotion = predict_emotion(image_array, 'Trained_Models/FER_model.sav')
        
        face_matrix = pd.DataFrame({'label' : pred_class, 
                                   'x_coordinate' : x_coordinate,
                                   'y_coordinate' : y_coordinate,
                                   'emotion' : pred_emotion})
        
        face_matrix = face_matrix[face_matrix['label'] != 'other']
        face_matrix.reset_index(drop=True,inplace=True)
        
    if showImage:
        class_img = cv2.imread(inputPath)
        for idx, (x,y,w,h) in enumerate(opencv_coordinates):
            if pred_class[idx] != 'other':
                cv2.rectangle(class_img,(x,y),(x+w,y+h),(255,0,0),3)
                cv2.putText(class_img, pred_class[idx],(x,y),cv2.FONT_HERSHEY_SIMPLEX,2,(0,255,0),4)
            else:
                cv2.rectangle(class_img,(x,y),(x+w,y+h),(0,0,255),3)
                cv2.putText(class_img, pred_class[idx],(x,y),cv2.FONT_HERSHEY_SIMPLEX,2,(0,255,0),4)
        pp.show_image(pp.resize_image(class_img, 1000))
    
    return face_matrix

# Please enter the input_path to the test image then:
    # select valid featureType name: 'Flatten', 'HOG', 'ORB'
    # select valid classifierName: 'CNN', 'SVM', 'Logistic_Regression', or 'Naive_Bayes'
    # can set showImage to False if only want dataframe

#RecogniseFace('Group_Images/IMG_8241.jpg', featureType='HOG',
#              classifierName='Logistic_Regression', showImage=True)


