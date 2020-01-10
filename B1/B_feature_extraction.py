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
        all_image = []
        all_features = []
        all_faceshape_labels = []
        all_eyecolour_labels = []
        eyecolourLabels = []
        faceshapeLabels = []
        for img_path in image_paths:
            img_name= img_path.split('/')[-1]
            # load image
            img = image.img_to_array(
                image.load_img(img_path,
                               target_size=target_size,
                               interpolation='bicubic'))
            features, resized_image = run_dlib_shape(img)
            if features is not None:
                # resize image to 128
                # change y shape from (,1) to (,5)
                print(resized_image.shape)
                all_image.append(resized_image)
                all_features.append(features)
                faceshapeLabel = np.zeros(5)
                faceshapeLabel[faceshape_labels[img_name]] = 1.0
                faceshapeLabels.append(faceshapeLabel)
                all_faceshape_labels.append(faceshape_labels[img_name])
                eyecolourLabel = np.zeros(5)
                eyecolourLabel[eyecolour_labels[img_name]] = 1.0
                eyecolourLabels.append(eyecolourLabel)
                all_eyecolour_labels.append(eyecolour_labels[img_name])
                if(len(all_faceshape_labels)%100 == 0):
                    print("Extracting image features {0}00/{1}".format(len(all_faceshape_labels)//100,len(faceshape_labels)))
    all_image = np.array(all_image)
    landmark_features = np.array(all_features)
    faceshapeLabels = np.array(faceshapeLabels)
    all_faceshape_labels = np.array(all_faceshape_labels)
    eyecolourLabels = np.array(eyecolourLabels)
    all_eyecolour_labels = np.array(all_eyecolour_labels)
    return all_image, landmark_features, all_faceshape_labels, all_eyecolour_labels, faceshapeLabels, eyecolourLabels

# 68points
# B1 for jaw
# x coordinate | y coordinate | face shape
# B2 for eye colour
#
