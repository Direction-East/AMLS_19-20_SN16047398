
##################### load additional testset ###############

add_testset_B_dir_path = os.path.join(basedir,'dataset_test/cartoon_set_test/img')
add_testset_B_labels_path = 'dataset_test/cartoon_set_test/labels.csv'
image_size = 500
num_channels = 3
add_testset_image_paths = [os.path.join(add_testset_B_dir_path,l) for l in os.listdir(add_testset_B_dir_path)]
labels_file = open(os.path.join(basedir, add_testset_B_labels_path), 'r')
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
    resized_image = cv2.resize(img, (128,128),0,0, cv2.INTER_LINEAR)
    test_all_image.append(resized_image)
    test_faceshapeLabel = np.zeros(5)
    test_faceshapeLabel[add_testset_faceshape_labels[img_name]] = 1.0
    test_faceshapeLabels.append(test_faceshapeLabel)
    test_eyecolourLabel = np.zeros(5)
    test_eyecolourLabel[add_testset_eyecolour_labels[img_name]] = 1.0
    test_eyecolourLabels.append(test_eyecolourLabel)
    if(len(test_faceshapeLabels)%100 == 0):
        print("Extracting image features {0}00/{1}".format(len(test_faceshapeLabels)//100,len(add_testset_faceshape_labels)))
    break
test_all_image = np.array(all_image)
test_landmark_features = np.array(all_features)
test_faceshapeLabels = np.array(faceshapeLabels)
test_eyecolourLabels = np.array(eyecolourLabels)

############# load saved tf model #####################


############# prediction and test #####################


############# show result #############################
