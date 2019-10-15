import os
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from sklearn.externals import joblib

import Preprocessing as pp

import numpy as np
import pandas as pd
import cv2

import keras
from keras.engine import  Model
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras import models, layers, optimizers

from keras.preprocessing.image import ImageDataGenerator
from keras.applications import VGG16 
from keras.callbacks import EarlyStopping, ModelCheckpoint

#from tensorflow.python.client import device_lib
#print(device_lib.list_local_devices())

## Shape and class inputs
img_classes = np.loadtxt('img_classes.txt', dtype='str')
nb_class = len(img_classes)
img_input_size = 90
img_input_shape = (img_input_size,img_input_size,3)
seed=34

# %% Find Faces in Group Image and Classify

def extract_faces(input_path, output_path=None, save=False, show=False):
    """Find and extract the faces in an image, can alternatively just show bounding boxes over the images.
    Has two levels of Haar filters one with a lower resolution pass and one with higher in order to remove 
    false positives."""
    
    img = cv2.imread(input_path)
    gray = pp.grayscale(img);
    
    face_cascade = cv2.CascadeClassifier('/Haar_Filters/haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray,1.05,4)
    
    if save:
        for i,(x,y,w,h) in enumerate(faces):
            face = img[y:y+h, x:x+w]
            if len(face_cascade.detectMultiScale(face,1.05,6)) == 1:
                cv2.imwrite(output_path+'/'+str(i)+'.jpeg', face)
    
    if show:
        for i,(x,y,w,h) in enumerate(faces):
            face = img[y:y+h, x:x+w]
            if len(face_cascade.detectMultiScale(face,1.05,6)) == 1:
                cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),3)
            else:
                cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),3)
        pp.show_image(pp.resize_image(img, 1000))

# %% Extract images and build dataset
def build_dataframe(input_path, img_input_shape, conform_shape=False):
    """Extracts face images from input folder and builds dataframe reshaping if necessary"""
    number_classes = os.listdir(path=input_path)
    
    image_array = []
    class_label = []
    
    for folder in number_classes:
        image_array.extend([cv2.imread(os.path.join(input_path,folder,x)) for
                            x in os.listdir(os.path.join(input_path,folder))
                            if '.jpeg' in x or '.jpg' in x])
        class_label.extend([folder for x in os.listdir(os.path.join(input_path,folder)) if
                            '.jpeg' in x or '.jpg' in x])
    
    if conform_shape:
        if max([len(img) for img in image_array]) > img_input_shape[0]:
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
    image_array = [x.astype('uint8') for x in image_array]
    
    return np.array(image_array), np.array(class_label)

def standardize(X_train_input, X_test_input):
    """Performs standard scaling on the X_inputs without data leakage from test set"""
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()
    sc.fit(X_train_input)

    X_train_std = sc.transform(X_train_input)
    X_test_std = sc.transform(X_test_input)
    
    return X_train_std, X_test_std

def subsample_dataframe(X_input, y_input, proportion=0.20):
    """Randomly subsamples the dataframe to reduce computational intensity"""
    assert len(X_input) == len(y_input), 'X and y arrays are not equal length'
    n_samples = len(X_input)
    np.random.seed(seed)
    random_index = np.random.choice(n_samples, int(n_samples * proportion), replace=False)
    X_input_sample, y_input_sample = X_input[random_index], y_input[random_index]
    return X_input_sample, y_input_sample

def create_train_test_sets(conform_shape=True, indi_proportion=0.50, incl_group_imgs=True):
    """Creates the train and test set subject to requirements in pipeline"""
    X_train_indi, y_train_indi = build_dataframe('Individual_Training_Images',
                                                 img_input_shape, conform_shape=conform_shape)
    X_test_indi, y_test_indi = build_dataframe('Individual_Test_Images',
                                               img_input_shape, conform_shape=conform_shape)
    
    X_train_group, y_train_group = build_dataframe('Group_Training_Images',
                                                       img_input_shape, conform_shape=conform_shape)
    X_test_group, y_test_group = build_dataframe('Group_Test_Images',
                                                     img_input_shape, conform_shape=conform_shape)
    
    X_train_indi, y_train_indi = subsample_dataframe(X_train_indi, y_train_indi,indi_proportion)
    
    if incl_group_imgs:
        X_train = np.concatenate([X_train_indi,X_train_group])
        y_train = np.concatenate([y_train_indi,y_train_group])
    else: 
        X_train = X_train_indi.copy()
        y_train = y_train_indi.copy()

    return X_train, y_train, X_test_indi, y_test_indi, X_test_group, y_test_group

# %% Feature Extraction Approaches and Classifiers
def flatten_array(X_input):
    """Flattens image matric into one dimensional pixel array"""
    X_input_flat = np.array([x.flatten() for x in X_input])
    return X_input_flat

def HOG_extractor(img):
    """Extracts HOG features from image
    source: https://stackoverflow.com/questions/27343614/opencv-hogdescriptor-compute-error"""
    try:
        img = pp.grayscale(img)
    except:
        pass
    winSize = (img.shape[0],img.shape[1])
    blockSize = (30,30)
    blockStride = (6,6)
    cellSize = (6,6)
    nbins = 9
    derivAperture = 1
    winSigma = 4.
    histogramNormType = 0
    L2HysThreshold = 2.0
    gammaCorrection = 0
    nlevels = 64
    hog = cv2.HOGDescriptor(winSize,blockSize,blockStride,cellSize,nbins,derivAperture,winSigma,
                            histogramNormType,L2HysThreshold,gammaCorrection,nlevels)
    hist = hog.compute(img)
    return hist

def extract_HOG_features(X_input):
    """Applies HOG feature extractor to array"""
    X_input_HOG = np.array([HOG_extractor(x).flatten() for x in X_input])
    return X_input_HOG

def ORB_feature_extractor(img, show = False):
    """Extracts ORB features from an image"""
    try:
        img = pp.grayscale(img)
    except:
        pass
    ORB = cv2.ORB_create(nfeatures = 60, scaleFactor = 1.2, nlevels = 8, patchSize = 15, 
                         edgeThreshold = 7, scoreType=cv2.ORB_FAST_SCORE)

    keypoints, description = ORB.detectAndCompute(img, None)

    if show:
        for point in keypoints:
            x,y = point.pt
            cv2.circle(img, (int(x), int(y)), 2,(0, 255, 255))
        pp.show_image(img)
    
    if description is None:
        description = np.zeros((1,ORB.descriptorSize()))
    
    return description

def kmeans_cluster(X_train_input, n_clusters=100):
    """Trains the kmeans model on the ORB descriptors for use in the VBOW"""
    from sklearn.cluster import MiniBatchKMeans
    image_descriptors = []
    [image_descriptors.extend(ORB_feature_extractor(img)) for img in X_train_input]
    image_descriptors = np.array(image_descriptors) 
    
    kmeans_model = MiniBatchKMeans(n_clusters=n_clusters, init_size=5*n_clusters,
                                   random_state=34, batch_size=128).fit(image_descriptors)
    
    return kmeans_model

def extract_ORB_features(X_input, kmeans_model, normalize=False):
    """Creates a visual bag of words using the extracted ORB descriptors and kmeans model"""
    kmeans_array = np.array([kmeans_model.predict(ORB_feature_extractor(img)) for img in X_input])
    bovw_dict = np.zeros((len(X_input),kmeans_model.n_clusters),dtype=('uint8'))
    for idx, arr in enumerate(kmeans_array):
        for j in arr:
            bovw_dict[idx,j] +=1
    
    if normalize:
        bovw_dict = np.divide(bovw_dict, bovw_dict.sum(axis=1).reshape(-1,1)).astype('float32')
    return bovw_dict

def train_linear_SVM(X_train_input, y_train_input, C=1):
    """Trains a linear SVM model"""
    from sklearn.svm import SVC
    svc_clf = SVC(kernel='linear', probability=True, C=C)
    svc_clf.fit(X_train_input, y_train_input)
    return svc_clf

def train_logistic_regression(X_train_input, y_train_input, C=1):
    """Trains a logistic regression model"""
    from sklearn.linear_model import LogisticRegression
    logr_clf = LogisticRegression(C=C)
    logr_clf.fit(X_train_input, y_train_input)
    return logr_clf

def train_naive_bayes(X_train_input, y_train_input):
    """Trains a gaussian naive bayes model"""
    from sklearn.naive_bayes import GaussianNB
    nb_clf = GaussianNB()
    nb_clf.fit(X_train_input, y_train_input)
    return nb_clf

def evaluate_model(model, X_test_input, y_test_input):
    """Evaluates the selected trained model on the holdout test set in terms of accuracy"""
    pred_class = [model.classes_[i] for i in model.predict_proba(X_test_input).argmax(axis=-1)]
    pred_accuracy = np.sum(np.array(y_test_input)==np.array(pred_class))/len(pred_class)
    return pred_class, pred_accuracy

def save_model(model, model_name):
    """Saves the model as a .sav file using joblib"""
    if os.path.isfile(model_name):
        print('Error: File already exists - please change name or remove conflicting file')
    else:
        joblib.dump(model, model_name)
    
def load_model(model_name):
    """Loads the trained model in joblib format"""
    model = joblib.load(model_name)
    return model

def train_model_pipeline(conform_shape=True, indi_proportion=0.50, incl_group_imgs=True,
                         feature_extractor=flatten_array, model=train_logistic_regression):
    """Creates a dataframe given requirements, trains, and evaluates the model""" 
    # Create dataframe subject to feature extractor requirements
    X_train, y_train, X_test_indi, y_test_indi, X_test_group, y_test_group = \
        create_train_test_sets(conform_shape=conform_shape, indi_proportion=indi_proportion, 
                               incl_group_imgs=incl_group_imgs)
    
    # Extract features
    if feature_extractor == extract_ORB_features:
        if os.path.isfile('Trained_Models/Kmeans_model.sav'):
            kmeans_model = load_model('Trained_Models/Kmeans_model.sav')
        else:
            kmeans_model = kmeans_cluster(X_train, 500)
        X_train = feature_extractor(X_train, kmeans_model, normalize = False)
        X_test_indi = feature_extractor(X_test_indi, kmeans_model, normalize = False)
        X_test_group = feature_extractor(X_test_group, kmeans_model, normalize = False)

    else:
        X_train = feature_extractor(X_train)
        X_test_indi = feature_extractor(X_test_indi)
        X_test_group = feature_extractor(X_test_group)
    
    # Train model on flattened array (no feature extraction)
    trained_model = model(X_train, y_train)
    
    indi_pred_class, indi_accuracy = evaluate_model(trained_model, X_test_indi, y_test_indi)
    group_pred_class, group_accuracy = evaluate_model(trained_model, X_test_group, y_test_group)
    
    return trained_model, indi_pred_class, indi_accuracy, group_pred_class, group_accuracy

#trained_model, _, indi_accuracy, group_pred_class, group_accuracy = train_model_pipeline(conform_shape = True,
#                                                             indi_proportion = 0.60,
#                                                             incl_group_imgs = True,
#                                                             feature_extractor = extract_HOG_features,
#                                                             model = train_logistic_regression)
#print(indi_accuracy)
#print(group_accuracy)
#
#save_model(trained_model, 'Trained_Models/Naive_Bayes_ORB.sav')

#save_model(kmeans_model, 'Trained_Models/Kmeans_model.sav')
    
#cm = pd.crosstab(y_test_group, np.array(group_pred_class), rownames=['True'], colnames=['Predicted'], margins=True)
#cm.to_csv('confusion_matrix.csv')
#

# %% CNN Model

def instantiate_VGG_model(img_input_shape):
    """Loads the pretrained VGG model and prepares for pretraining"""
    # Load the VGG model
    vgg_conv = VGG16(weights='imagenet', include_top=False, input_shape=img_input_shape)
    
    # Freeze the layers except the last 4 layers
    for layer in vgg_conv.layers[:-4]:
        layer.trainable = False
    
    # Create the model
    model = models.Sequential()
    model.add(vgg_conv)
     
    # Add new layers
    model.add(layers.Flatten())
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dropout(0.25))
    model.add(layers.Dense(nb_class, activation='softmax'))
    
    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizers.RMSprop(lr=1e-4),
                  metrics=['acc'])
    return model

def train_VGG_model(X_train_input, y_train_input, model, epochs=30, batch_size=32, patience=10):
    """Trains the VGG model as a generator"""
    from sklearn.preprocessing import LabelBinarizer   
    # One-hot encode target
    lb = LabelBinarizer()
    lb.fit(y_train_input)
    
    y_train_onehot = lb.transform(y_train_input)
    
    # Augmentation step
    train_batches = ImageDataGenerator().flow(X_train_input, y_train_onehot,
                                       batch_size=batch_size,shuffle=True)
    
    early_stopping_callback = EarlyStopping(monitor='acc', patience=patience, restore_best_weights=True)
    
    model.fit_generator(train_batches, steps_per_epoch=len(X_train_input)//batch_size,
                        epochs=epochs,verbose=2, callbacks=[early_stopping_callback])
    return model, lb

def evaluate_VGG_model(model, X_test_input, y_test_input, lb):
    """Evaluates the VGG model on the test set"""
    test_batches = ImageDataGenerator().flow(X_test_input, y_test_input,
                                  batch_size=1,shuffle=False)

    predictions = model.predict_generator(test_batches, steps = len(y_test_input))
    pred_class = np.array([lb.classes_[i] for i in predictions.argmax(axis=-1)])
    pred_accuracy = np.sum(np.array(y_test_input)==np.array(pred_class))/len(pred_class)
    
    return pred_class, pred_accuracy

def train_VGG_model_pipeline(conform_shape=True, indi_proportion=0.50, incl_group_imgs=True):
    """Pipeline which creates, trains, and evaluates the model"""
    # Create dataframe subject to feature extractor requirements
    X_train, y_train, X_test_indi, y_test_indi, X_test_group, y_test_group = \
        create_train_test_sets(conform_shape=conform_shape, indi_proportion=indi_proportion, 
                               incl_group_imgs=incl_group_imgs)
    
    VGG_model = instantiate_VGG_model(img_input_shape)
    VGG_model, lb = train_VGG_model(X_train, y_train, VGG_model, 60, 32, 10)
    
    indi_pred_class, indi_accuracy = evaluate_VGG_model(VGG_model, X_test_indi, y_test_indi, lb)    
    group_pred_class, group_accuracy = evaluate_VGG_model(VGG_model, X_test_group, y_test_group, lb)

    return VGG_model, indi_pred_class, indi_accuracy, group_pred_class, group_accuracy

#VGG_model, _, indi_accuracy, _, group_accuracy = train_VGG_model_pipeline(conform_shape = True,
#                                                             indi_proportion = 0.60,
#                                                             incl_group_imgs = False)
#print(indi_accuracy)
#print(group_accuracy)
##
#VGG_model.save('Trained_Models/VGG_model.h5')

#%% Emotion detection

def train_emotion_detection_model(input_path, data_proportion=0.15):
    """Loads, reformats, trains and evaluates the facial expression recognition model"""
    # Load data (https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data)
    emotion_df = pd.read_csv(input_path)
    text_labels = {0:'Angry', 1:'Disgust', 2:'Fear', 3:'Happy', 4:'Sad', 5:'Surprised', 6:'Neutral'}
    
    # Select only valid emotions per CW 
    valid_labels = [0, 3, 4, 5]
    emotion_df = emotion_df.loc[emotion_df['emotion'].isin(valid_labels)]
    
    # Transform data from strings to int and reshape 
    image_array = np.array([x.split(' ') for x in emotion_df['pixels']],dtype='uint8')
    image_array = image_array.reshape((-1,48,48))
    target_labels = emotion_df['emotion'].values
    
    # Create Train / Test split
    X_values, y_values = subsample_dataframe(image_array, target_labels, data_proportion)
    
    # Reshape data
    X_values = np.array([pp.resize_image(x,img_input_shape[0]) for x in X_values])
    
    train_test_split = 0.70
    index_val = int(len(X_values)*train_test_split)
    
    X_train_FER, y_train_FER = X_values[:index_val], y_values[:index_val]
    X_test_FER, y_test_FER = X_values[index_val+1:], y_values[index_val+1:]
    
    X_train_FER = extract_HOG_features(X_train_FER)
    X_test_FER = extract_HOG_features(X_test_FER)
    
    trained_model = train_logistic_regression(X_train_FER, y_train_FER)
    
    FER_pred_class, FER_accuracy = evaluate_model(trained_model, X_test_FER, y_test_FER)
    label_predictions = np.vectorize(text_labels.get)(FER_pred_class)
    
    return trained_model, label_predictions, FER_accuracy

#FER_model, _, FER_accuracy = train_emotion_detection_model('Facial_Expression/fer2013.csv', 0.15)
#
#print(FER_accuracy)
#save_model(FER_model, 'FER_model.sav')

# %% Autoencoder
    
from keras.layers import Input, Dense
from keras.models import Model

def simple_autoencoder(X_train_input, X_test_input, n_components = 100):
    """Creates, fits, and transforms the inputs using a single hidden layer autoencoder"""
    ncol = X_train_input.shape[1]
    input_dim = Input(shape = (ncol,))
    
    # Define the number of encoder dimensions
    encoding_dim = n_components
    
    # Define the encoder layer
    encoded = Dense(encoding_dim, activation = 'relu')(input_dim)
    
    # Define the decoder layer
    decoded = Dense(ncol, activation = 'tanh')(encoded)
    
    # Combine the encoder and decoder into a model
    autoencoder = Model(inputs = input_dim, outputs = decoded)
    
    # Configure and train the autoencoder
    autoencoder.compile(optimizer = 'adam', loss = 'mse')
    autoencoder.fit(X_train_input, X_train_input, epochs = 50, batch_size = 128, shuffle = True,
                    validation_data = (X_test_input, X_test_input),verbose = 1)
    
    # Use the encoder to extract the reduced dimension from the autoencoder
    encoder = Model(inputs = input_dim, outputs = encoded)
    
    X_train_output = encoder.predict(X_train_input)
    X_test_output = encoder.predict(X_test_input)
    
    return X_train_output, X_test_output


def deep_autoencoder(X_train_input, X_test_input, encoding_dim = 20):
    """Creates, fits, and transforms the inputs using a multiple hidden layer autoencoder
        Source: https://ramhiser.com/post/2018-05-14-autoencoders-with-keras/"""
    input_dim = X_train_input.shape[1]
    
    autoencoder = Sequential()
    
    # Encoder Layers
    autoencoder.add(Dense(4 * encoding_dim, input_shape=(input_dim,), activation='relu'))
    autoencoder.add(Dense(2 * encoding_dim, activation='relu'))
    autoencoder.add(Dense(encoding_dim, activation='relu'))
    
    # Decoder Layers
    autoencoder.add(Dense(2 * encoding_dim, activation='relu'))
    autoencoder.add(Dense(4 * encoding_dim, activation='relu'))
    autoencoder.add(Dense(input_dim, activation='sigmoid'))
    
    autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
    autoencoder.fit(X_train_input, X_train_input,
                    epochs=50,
                    batch_size=256,
                    validation_data=(X_test_input, X_test_input))
    
    input_img = Input(shape=(input_dim,))
    encoder_layer1 = autoencoder.layers[0]
    encoder_layer2 = autoencoder.layers[1]
    encoder_layer3 = autoencoder.layers[2]
    encoder = Model(input_img, encoder_layer3(encoder_layer2(encoder_layer1(input_img))))
    
    X_train_output = encoder.predict(X_train_input)
    X_test_output = encoder.predict(X_test_input)
    
    return X_train_output, X_test_output

#X_train_output, X_test_output = deep_autoencoder(X_train, X_test, 25)

# LightGBM
#import lightgbm as lgb
#
#lgbm_clf = lgb.LGBMClassifier(num_leaves=7, n_estimators=300, objective='multiclass',
#                                  random_state=34).fit(X_train_output,y_train)
#
#val_preds = [img_classes[i] for i in lgbm_clf.predict_proba(X_val).argmax(axis=-1)]
#test_preds = [img_classes[i] for i in lgbm_clf.predict_proba(X_test_output).argmax(axis=-1)]
#
#val_accuracy = np.sum(np.array(y_val)==np.array(val_preds))/len(y_val)
#test_accuracy = np.sum(np.array(y_test)==np.array(test_preds))/len(y_test)
#print(val_accuracy)
#print(test_accuracy)

# %% Extract  group images into folders, rename, and then merge folders

def classify_face(img, HOG_model):
    """Classifies face for labelling the group images"""
    img = pp.resize_image(img, img_input_size)
    
    # Ensure shape matches exactly
    shape_delta = img_input_shape[0] - img.shape[0]
    if shape_delta > 0:
        new_row = np.random.randint(0,255,[shape_delta,img_input_shape[1],img_input_shape[2]],dtype='uint8')
        img = np.vstack([img, new_row])
    
    elif shape_delta < 0:
        img = img[:img_input_shape[0],:,:] 
    
    HOG_img = HOG_extractor(img).flatten()
    class_pred = img_classes[HOG_model.predict_proba([HOG_img]).argmax(axis=-1)[0]]
    return class_pred

def label_training_data(input_path, output_path):
    """Labels the group training data"""
    import shutil
    image_files = [file for file in os.listdir(path=input_path) if '.JPG' in file or '.jpeg' in file]
    
    for file in image_files:
        file_input_path = os.path.join(input_path,file)
        
        img = cv2.imread(file_input_path)
        
        file_output_path = os.path.join(output_path, classify_face(img))
        
        try:
            os.makedirs(file_output_path)
        except FileExistsError:
            # directory already exists
            pass
        shutil.move(file_input_path, file_output_path)

#label_training_data('Group_Test_Images/Level16','Group_Test_Images/Level16')

def rename_images():
    """Rename all images extracted from group photos"""
    grp_img_dir = os.listdir('Group_Training_Images')
    
    for grp_img_folder in grp_img_dir:
        image_folders = os.listdir('Group_Training_Images'+'/'+grp_img_folder)
        
        for img_label in image_folders:
            image_path = 'Group_Training_Images'+'/'+grp_img_folder+'/'+img_label
            
            original_file_names = os.listdir(image_path)
            
            if len(original_file_names) > 1:
                for idx, img in enumerate(os.listdir(image_path)):
                    assert '.jpeg' in img or '.jpg' in img, img +' incorrect format'
                    new_name = img_label+'_'+grp_img_folder+'_'+str(idx+1)+'.jpeg'
                    os.rename(image_path+'/'+img, image_path+'/'+ new_name)
            else:
                assert ('.jpeg' in original_file_names[0] or 
                        '.jpg' in original_file_names[0]), original_file_names[0] +' incorrect format'
                new_name = img_label+'_'+grp_img_folder+'.jpeg'
                os.rename(image_path+'/'+original_file_names[0], image_path+'/'+ new_name)

def merge_folders():
    """Merge folders so images are all in the same folder number"""
    from shutil import copyfile
    # Merge all folders into main folder
    grp_img_dir = os.listdir('Group_Test_Images')
    
    for grp_img_folder in grp_img_dir:
        image_folders = os.listdir('Group_Test_Images'+'/'+grp_img_folder)
        
        for img_label in image_folders:
            new_directory = 'Group_Test_Images'+'/'+img_label
            
            try:
                os.makedirs(new_directory)
            except FileExistsError:
                # directory already exists
                pass
            
            file_names = os.listdir('Group_Test_Images'+'/'+grp_img_folder+'/'+img_label)
            
            for file in file_names:
                copyfile('Group_Test_Images'+'/'+grp_img_folder+'/'+img_label+'/'+file, new_directory+'/'+file)
