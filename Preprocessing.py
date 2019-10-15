import cloudconvert

import cv2
import pytesseract
#pytesseract.pytesseract.tesseract_cmd = r"C:\Users\AGB\AppData\Local\Tesseract-OCR\tesseract.exe"


import numpy as np
import os


# Define OpenCV helper functions

def show_image(img):
    """Creates a new window showing the image"""
    cv2.imshow('image',img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
def resize_image(image, width = None, height = None, inter = cv2.INTER_AREA):
    """Resizes the input without distortion 
        # Function source: https://stackoverflow.com/questions/44650888/resize-an-image-without-distortion-opencv """
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv2.resize(image, dim, interpolation = inter)
    return resized

def grayscale(img):
    """Converts the image to grayscale"""
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img

def variance_of_laplacian(img): 
    """Returns the variance of the laplacian of image to determine image blur
       based on: https://www.pyimagesearch.com/2015/09/07/blur-detection-with-opencv/"""
    # compute the Laplacian of the image and then return the focus
    # measure, which is simply the variance of the Laplacian
    return cv2.Laplacian(img, cv2.CV_64F).var()

# Convert HEIC Files to JPEG

#API Key
API_KEY = 'YOUR_API_KEY'
api = cloudconvert.Api(API_KEY)

def HEIC_conversion(input_path, output_path):
    """Converts all .HEIC files in a given directory to jpg and places 
    them in the output path under the same file name"""
    HEIC_files = [file[:-5] for file in os.listdir(path=input_path) if '.HEIC' in file]

    for file in HEIC_files:
        process = api.convert({
            'inputformat': 'HEIC',
            'outputformat': 'jpg',
            'input': 'upload',
            'file': open(input_path+file+'.HEIC', 'rb')
        })
        process.wait() # wait until conversion finished
        process.download(output_path+file+'.jpg') # download output file

# %% Detect face and detect number in image

def detect_face(img, frontal_horiz_adj = 0.2, profile_horiz_adj = 0.35):
    """Takes color image and returns face image as well as coordinates for top, left and right
    frontal_horiz_adj is the amount of padding from the midpoint if cascade matches from the front
    profile_horiz_adj is the amount of padding from the midpoint if cascade matches from the side"""
    
    try:
        gray = grayscale(img)
    except:
        gray = img.copy()
    
    face_cascade = cv2.CascadeClassifier('Haar_Filters/haarcascade_frontalface_default.xml')
    face = face_cascade.detectMultiScale(gray,1.05,20)

    if len(face)==1:
        (x,y,w,h) = face[0]
        face = img[y:y+h, x:x+w]
        horizontal_adj = frontal_horiz_adj 

    elif len(face)==0:
        face_cascade = cv2.CascadeClassifier('Haar_Filters/haarcascade_profileface.xml')
        face = face_cascade.detectMultiScale(gray,1.05,20)

        if len(face)==1:
            (x,y,w,h) = face[0]
            face = img[y:y+h, x:x+w]
            horizontal_adj = profile_horiz_adj
            
        elif len(face)==0:
            face_cascade = cv2.CascadeClassifier('Haar_Filters/haarcascade_frontalface_default.xml')
            face = face_cascade.detectMultiScale(gray)
            if len(face)>0:           
                (x,y,w,h) = face[0]
                face = img[y:y+h, x:x+w]
                horizontal_adj = frontal_horiz_adj
            else:
                print("Unable to locate face")
                return
        else:
            print("Unable to locate face")
            return
    else:
        print("Unable to locate face")
        return
    
    top = y+h
    left = int(x*(1-horizontal_adj))
    right = int((x+w)*(1+horizontal_adj))
    
    return face, top, left, right

def cut_image(input_path, image_size=500, frontal_horiz_adj = 0.2, profile_horiz_adj = 0.35):
    """Returns cropped image below the face and sides"""
    img = cv2.imread(input_path)
    img = resize_image(img, image_size)
    face, top, left, right = detect_face(img,frontal_horiz_adj,profile_horiz_adj)
    
    img = img[top:,left:right]
    return img

def detect_blobs(img, blur=23, min_area = 10, additional_checks = True, show_keypoints = False):
    """Returns x,y,size coordinates of all detected blobs"""
    # Setup SimpleBlobDetector parameters
    img = cv2.GaussianBlur(img,(blur,blur), 0)
    params = cv2.SimpleBlobDetector_Params()

    # Filter by Area
    params.filterByArea = True
    params.minArea = min_area
    
    if additional_checks:
        # Filter by Inertia
        params.filterByInertia = True
        params.minInertiaRatio = 0.4

        # Filter by Convexity
        params.filterByConvexity = True
        params.minConvexity = 0.4

    # Create a detector with the parameters
    detector = cv2.SimpleBlobDetector_create(params)

    # Detect blobs
    keypoints = detector.detect(img)
    keypoints = [(kp.pt[0], kp.pt[1], kp.size) for kp in keypoints]
    
    # Show keypoints
    if show_keypoints:
        for x,y,s in keypoints:
            cv2.circle(img, (int(x), int(y)), int(s),(0, 255, 255))
        show_image(img)
    
    return keypoints

def check_blob(img, x, y, vertical_adj = 0.1, horizontal_adj = 0.2,
               gray_thresh = 120, border_intensity_thresh = 250):
    """Checks whether blob is likely to be the number based on its surrounding area"""
    top = int(max(y-(img.shape[0]*vertical_adj),0))
    bottom = int(min(y+(img.shape[0]*vertical_adj),img.shape[0]-1))
    left = int(max(x-(img.shape[1]*horizontal_adj),0))
    right = int(min(x+(img.shape[1]*horizontal_adj),img.shape[1]-1))

    ret,thresh = cv2.threshold(grayscale(img),gray_thresh,255,0)
    
    top_line = np.array([thresh[top,p] for p in range(left,right)])
    bottom_line = np.array([thresh[bottom,p] for p in range(left,right)])
    left_line = np.array([thresh[p,left] for p in range(top,bottom)])
    right_line = np.array([thresh[p,right] for p in range(top,bottom)])
    
    avg_border_intensity = np.mean([top_line.mean(), bottom_line.mean(),
                                    left_line.mean(), right_line.mean()])
    
    if avg_border_intensity > border_intensity_thresh:
        img = img[top:bottom,left:right]
        return img, top, left
    
    img = img[top:bottom,left:right]
    return img, top, left

def remove_close_blobs(detected_blobs, min_distance = 20):
    """Removes blobs which are too close together as are likely duplicated numbers"""
    if len(detected_blobs)>1:
        for index,(img, x1, y1) in enumerate(detected_blobs):
            for index,(img, x2, y2) in enumerate(detected_blobs):
                euclidean_distance = ((x1-x2)**2+(y1-y2)**2)**.5
                if 0 < euclidean_distance <= min_distance:
                    detected_blobs.pop(index)
        number_img = [img for (img, x1, y1) in detected_blobs]
    else:
        number_img = [detected_blobs[0][0]]
    return number_img

def detect_number(img):
    """Detects the number in the image using Pytesseract"""
    img = grayscale(img)
    
    kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    img = cv2.filter2D(img, -1, kernel)
    img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    
    number = pytesseract.image_to_data(
            img, config='--psm 8 --oem 3 -c tessedit_char_whitelist=0123456789',
            output_type='data.frame') #config='outputbase digits'
    number, confidence = number[['text','conf']].dropna().values[0]
    if number != ' ':
        return number, confidence
    else:
        return None, 0

def detectNum(input_path, detect_multiple=False, debug = False, video_input=False):
    """Detects number from raw image by applying processing pipeline functions"""
    
    if video_input:
        img = input_path.copy()
        
        if detect_multiple:
            img = resize_image(img,500)
        else:
            try:
                img = resize_image(img, 500)
                _, top, left, right = detect_face(img,0.2,0.35)
                img = img[top:,left:right]
                
            except:
                img = resize_image(img, 500)
    
    else:
        if detect_multiple:
            img = resize_image(cv2.imread(input_path),500)
        else:
            try:
                img = cut_image(input_path)
            except:
                img = resize_image(cv2.imread(input_path),500)
    
    try:
        keypoints = detect_blobs(img,blur=31,min_area=10,additional_checks=True)
        if len(keypoints) == 0:
            keypoints = detect_blobs(img,blur=35,min_area=10,additional_checks=False)
    
        blobs = [check_blob(img, x=kp[0], y=kp[1], vertical_adj=0.05, horizontal_adj=0.1) for kp in keypoints]
        blobs = [x for x in blobs if x is not None]
        number_img = remove_close_blobs(blobs,30)
    except:
        print("Unable to detect blobs: ", input_path[-13:])
        return       
    
    try:
        detected_numbers = np.array([detect_number(n) for n in number_img])
        if detect_multiple:
            return detected_numbers[:,0]
        else:
            predicted_number = detected_numbers[np.argmax([n[1] for n in detected_numbers])][0]
            return predicted_number
    except:
        print("Unable to detect number: ", input_path[-13:])
        if debug:
            print("File: ",input_path[-13:])
            print("keypoints: ",len(keypoints))
            print("blobs: ",len(blobs))
            print("number blobs: ",len(blobs))
            print("detected_numbers: ", detected_numbers)
            print("predicted_number: ", predicted_number)

def save_face(img, output_path, image_size=90, min_blur_thresh=150):
    """Helper function for saving the extracted faces.
    Note: output_path must include the file name with extension"""
    img = resize_image(img,500)
    try:
        face,_,_,_ = detect_face(img)
        face = resize_image(face,image_size)
        if variance_of_laplacian(face) > min_blur_thresh:
            cv2.imwrite(output_path, face)
    except:
        print('Unable to save face')
        return

def extract_faces_to_img_folder(input_path, output_path, image_size=90):
    """Extracts the faces from the individual images to the appropriate 
    folder based on image detection"""
    files_to_extract = [file for file in os.listdir(path=input_path) if '.jpeg' in file or '.jpg' in file]
    failed_files = []
    for file in files_to_extract:
        image_file_path = os.path.join(input_path,file)
        img = cv2.imread(image_file_path)
        try:
            number_folder = int(float(detectNum(image_file_path)))
        except:
            try:
                save_face(img,os.path.join(output_path,'other/',file),image_size)
            except:
                failed_files.append(file)
                pass
        try:
            if number_folder < 100:
                os.makedirs(os.path.join(output_path,str(number_folder)))
            else:
                save_face(img,os.path.join(output_path,'other/',file),image_size)
        except FileExistsError:
            pass
        
        try:
            save_face(img, os.path.join(output_path, str(number_folder),file), image_size)
        except:
            failed_files.append(file)
            pass
    return failed_files

def show_face(img):
    """Detects and shows the image from an individual picture"""
    img = resize_image(img, 500)
    face,_,_,_ = detect_face(img)
    show_image(face)

# Extract Images From Videos
def extract_videos(input_path, output_path, n_frames=10, blur_thresh=150, only_face=True):
    """ Extracts every nth frame from video file and exports face as jpeg"""
    try:
        os.makedirs(output_path)
    except FileExistsError:
        # directory already exists
        pass
    
    video_files = [file for file in os.listdir(path=input_path) if '.mov' in file or '.mp4' in file]

    for file in video_files:   
        vidcap = cv2.VideoCapture(os.path.join(input_path,file))
        success,img = vidcap.read()

        count = 0
        while success:
            if count % n_frames == 0:
                if '.mov' in file:
                    (rows, cols) = img.shape[:2]
                    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), 270, 1) 
                    img = cv2.warpAffine(img, M, (cols, rows))
                if only_face:
                    save_face(img,os.path.join(
                            output_path, "{}_f{}.jpeg".format(file[:-4],count)),90,blur_thresh)
                else:
                    cv2.imwrite(os.path.join(output_path, "{}_f{}.jpeg".format(file[:-4],count)), img)
            success,img = vidcap.read()
            count += 1

# %% Apply functions
            
# Convert HEIC to JPEG
#HEIC_conversion('Group_Images/OriginalHEIC/','Group_Images/OriginalHEIC/')

# Detect number in image
#detectNum('Individual_Images/Camera1/IMG_3493.jpeg',debug=False)

# Extracts faces to numbered folders based on OCR
#failed_files = extract_faces_to_img_folder(input_path = 'Individual_Images/Camera1/',
#                            output_path = 'Individual_Images/', image_size=90)
    

# Extract faces from videos to folders
# For individual person
#extract_videos(input_path = 'Individual_Images/Camera2/mov/4/',output_path =
#     'Individual_Images/4/',n_frames=5,blur_thresh=150,only_face=True)

# For all videos in folders
#video_folder = os.listdir('Individual_Images/Camera3/mov/')
#
#[extract_videos(input_path = 'Individual_Images/Camera3/mov/'+folder+'/',
#                output_path ='Test_Images/'+folder+'/',
#                n_frames=19,blur_thresh=120,only_face=True) for folder in video_folder]


    
    