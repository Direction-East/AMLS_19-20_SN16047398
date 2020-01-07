import os
import numpy as np
from keras.preprocessing import image
import cv2
import dlib

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('./A1/shape_predictor_68_face_landmarks.dat')

def shape_to_np(shape, dtype="int"):
    # initialize the list of (x, y)-coordinates
    coords = np.zeros((shape.num_parts, 2), dtype=dtype)

    # loop over all facial landmarks and convert them
    # to a 2-tuple of (x, y)-coordinates
    for i in range(0, shape.num_parts):
        coords[i] = (shape.part(i).x, shape.part(i).y)

    # return the list of (x, y)-coordinates
    return coords

def rect_to_bb(rect):
    # take a bounding predicted by dlib and convert it
    # to the format (x, y, w, h) as we would normally do
    # with OpenCV
    x = rect.left()
    y = rect.top()
    w = rect.right() - x
    h = rect.bottom() - y

    # return a tuple of (x, y, w, h)
    return (x, y, w, h)

def run_dlib_shape(image):
    # in this function we load the image, detect the landmarks of the face, and then return the image and the landmarks
    # load the input image, resize it, and convert it to grayscale
    resized_image = image.astype('uint8')

    gray = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
    gray = gray.astype('uint8')

    # detect faces in the grayscale image
    rects = detector(gray, 1)
    num_faces = len(rects)

    if num_faces == 0:
        return None, resized_image

    face_areas = np.zeros((1, num_faces))
    face_shapes = np.zeros((136, num_faces), dtype=np.int64)

    # loop over the face detections
    for (i, rect) in enumerate(rects):
        # determine the facial landmarks for the face region, then
        # convert the facial landmark (x, y)-coordinates to a NumPy
        # array
        temp_shape = predictor(gray, rect)
        temp_shape = shape_to_np(temp_shape)

        # convert dlib's rectangle to a OpenCV-style bounding box
        # [i.e., (x, y, w, h)],
        #   (x, y, w, h) = face_utils.rect_to_bb(rect)
        (x, y, w, h) = rect_to_bb(rect)
        face_shapes[:, i] = np.reshape(temp_shape, [136])
        face_areas[0, i] = w * h
    # find largest face and keep
    dlibout = np.reshape(np.transpose(face_shapes[:, np.argmax(face_areas)]), [68, 2])

    return dlibout, resized_image

##########################
######## add some text reminding timing of extration ###########################
###########################
def A_extract_features_labels(basedir,images_dir, labels_filename):
    image_paths = [os.path.join(images_dir,l) for l in os.listdir(images_dir)]
    target_size = None
    labels_file = open(os.path.join(basedir, labels_filename), 'r')
    lines = labels_file.readlines()
    gender_labels = {line.split('\t')[1] : int(line.split('\t')[2]) for line in lines[1:]}
    smile_labels = {line.split('\t')[1] : int(line.split('\t')[3]) for line in lines[1:]}
    if os.path.isdir(images_dir):
        all_features = []
        all_gender_labels = []
        all_smile_labels = []
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
                all_gender_labels.append(gender_labels[img_name])
                all_smile_labels.append(smile_labels[img_name])
    landmark_features = np.array(all_features)
    all_gender_labels = (np.array(all_gender_labels) + 1)/2 # simply converts the -1 into 0, so male=0 and female=1
    all_smile_labels = (np.array(all_smile_labels) + 1)/2 # not smile -1 to 0, smile stays at 1
    return landmark_features, all_gender_labels, all_smile_labels

def A_get_tvt_dataset(basedir,images_dir, labels_filename):
    X, y_gender, y_smile = A_extract_features_labels(basedir,images_dir, labels_filename)
    Y_gender = np.array([y_gender, -(y_gender - 1)]).T
    Y_smile = np.array([y_smile, -(y_smile - 1)]).T
    return X, Y_gender, Y_smile