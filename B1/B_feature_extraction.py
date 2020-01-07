import os
import numpy as np
from keras.preprocessing import image
import cv2
import dlib
from A1.A_feature_extration import shape_to_np, rect_to_bb, run_dlib_shape

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('./B1/shape_predictor_68_face_landmarks.dat')

def B_extract_features_labels(basedir, images_dir, labels_filename):
    image_paths = [os.path.join(images_dir,l) for l in os.listdir(images_dir)]
    target_size = None
    labels_file = open(os.path.join(basedir, labels_filename), 'r')
    lines = labels_file.readlines()
    faceshape_labels = {line.split('\t')[3].strip('\n') : int(line.split('\t')[2]) for line in lines[1:]}
    eyecolour_labels = {line.split('\t')[3].strip('\n') : int(line.split('\t')[1]) for line in lines[1:]}
    if os.path.isdir(images_dir):
        all_features = []
        all_faceshape_labels = []
        all_eyecolour_labels = []
        for img_path in image_paths:
            img_name= img_path.split('/')[-1]
            # load image
            img = image.img_to_array(
                image.load_img(img_path,
                               target_size=target_size,
                               interpolation='bicubic'))
            features, _ = run_dlib_shape(img)
            if features is not None:
                all_features.append(features)
                all_faceshape_labels.append(faceshape_labels[img_name])
                all_eyecolour_labels.append(eyecolour_labels[img_name])
    landmark_features = np.array(all_features)
    all_faceshape_labels = np.array(all_faceshape_labels)
    all_eyecolour_labels = np.array(all_eyecolour_labels)
    return landmark_features, all_faceshape_labels, all_eyecolour_labels
