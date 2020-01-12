import os
import numpy as np
from keras.preprocessing import image
import cv2
import dlib
from A1.A_feature_extration import shape_to_np, rect_to_bb, run_dlib_shape

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('./B1/shape_predictor_68_face_landmarks.dat')

# ==========================================================
# future work: similar functionality, can be merged together
# ==========================================================

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

def B_load_additional_test_set(add_testset_B_dir_path, add_testset_B_labels_path):
    image_size = 500
    num_channels = 3
    add_testset_image_paths = [os.path.join(add_testset_B_dir_path,l) for l in os.listdir(add_testset_B_dir_path)]
    labels_file = open(add_testset_B_labels_path, 'r')
    lines = labels_file.readlines()
    add_testset_faceshape_labels = {line.split('\t')[3].strip('\n') : int(line.split('\t')[2]) for line in lines[1:]}
    add_testset_eyecolour_labels = {line.split('\t')[3].strip('\n') : int(line.split('\t')[1]) for line in lines[1:]}

    test_all_image = []
    test_eyecolourLabels = []
    test_faceshapeLabels = []
    for img_path in add_testset_image_paths:
        img_name= img_path.split('/')[-1]
        # load image
        img = image.img_to_array(
            image.load_img(img_path,
                           target_size=None,
                           interpolation='bicubic'))
    #     print(img.shape)
        resized_image = img
    #     resized_image = cv2.resize(img, (128,128),0,0, cv2.INTER_LINEAR)
        test_all_image.append(resized_image)
        test_faceshapeLabel = np.zeros(5)
        test_faceshapeLabel[add_testset_faceshape_labels[img_name]] = 1.0
        test_faceshapeLabels.append(test_faceshapeLabel)
        test_eyecolourLabel = np.zeros(5)
        test_eyecolourLabel[add_testset_eyecolour_labels[img_name]] = 1.0
        test_eyecolourLabels.append(test_eyecolourLabel)
        if(len(test_faceshapeLabels)%100 == 0):
            print("Extracting image features {0}00/{1}".format(len(test_faceshapeLabels)//100,len(add_testset_faceshape_labels)))
    test_all_image = np.array(test_all_image)
    test_all_image = test_all_image.astype('float32')
    test_all_image = np.multiply(test_all_image, 1.0/255.0)
    test_faceshapeLabels = np.array(test_faceshapeLabels)
    test_eyecolourLabels = np.array(test_eyecolourLabels)
    return test_all_image, test_faceshapeLabels, test_eyecolourLabels
