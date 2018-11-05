import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile
import glob

import scipy.misc as sm
from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image
import cv2
import time

# This is needed since the notebook is stored in the object_detection folder.
sys.path.append("..")
from utils import ops as utils_ops

if tf.__version__ < '1.4.0':
    raise ImportError('Please upgrade your tensorflow installation to v1.4.* or later!')

# This is needed to display the images.
# %matplotlib inline
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
from utils import label_map_util

from utils import visualization_utils as vis_util

# What model to download.
# MODEL_NAME = 'ssd_mobilenet_v1_coco_2017_11_17'
# MODEL_FILE = MODEL_NAME + '.tar.gz'
# DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'

# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = 'frozen_inference_graph_33638_truck.pb'
# frozen_inference_graph_33638_truck.pb
# frozen_inference_graph_bus_1017.pb
# frozen_inference_graph_truc_nohardexample_1028.pb
# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join('data', 'label_map.pbtxt')

NUM_CLASSES = 7
JILV = 0.6
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

## Loading label map
# Label maps map indices to category names, so that when our convolution network predicts `5`,
# we know that this corresponds to `airplane`.  Here we use internal utility functions,
# but anything that returns a dictionary mapping integers to appropriate string labels would be fine

label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES,
                                                            use_display_name=True)
category_index = label_map_util.create_category_index(categories)


def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape(
        (im_height, im_width, 3)).astype(np.uint8)


def load_image_into_numpy_array2(image):
    return np.array(image).reshape((480, 640, 3)).astype(np.uint8)


# Detection

# For the sake of simplicity we will use only 2 images:
# image1.jpg
# image2.jpg
# If you want to test the code with your images, just add path to the images to the TEST_IMAGE_PATHS.
PATH_TO_TEST_IMAGES_DIR = 'test_images/*.jpg'
# TEST_IMAGE_PATHS = [ os.path.join(PATH_TO_TEST_IMAGES_DIR)]
TEST_IMAGE_PATHS = glob.glob(PATH_TO_TEST_IMAGES_DIR)
# Size, in inches, of the output images.
IMAGE_SIZE = (12, 8)
'''for key in [
          'num_detections', 'detection_boxes', 'detection_scores',
          'detection_classes', 'detection_masks'
      ]:'''

session_config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)


def run_inference_for_single_image(image, graph):
    with graph.as_default():
        with tf.Session(config=session_config) as sess:
            with tf.device('/gpu:0'):
                # Get handles to input and output tensors
                start = time.time()
                ops = tf.get_default_graph().get_operations()
                all_tensor_names = {output.name for op in ops for output in op.outputs}
                tensor_dict = {}

                for key in [
                    'num_detections', 'detection_boxes', 'detection_scores',
                    'detection_classes', 'detection_masks'
                ]:
                    tensor_name = key + ':0'
                    if tensor_name in all_tensor_names:
                        tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(
                            tensor_name)
                if 'detection_masks' in tensor_dict:
                    # The following processing is only for single image
                    detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
                    detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])
                    # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
                    real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
                    detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
                    detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
                    detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
                        detection_masks, detection_boxes, image.shape[0], image.shape[1])
                    detection_masks_reframed = tf.cast(
                        tf.greater(detection_masks_reframed, JILV), tf.uint8)
                    # Follow the convention by adding back the batch dimension
                    tensor_dict['detection_masks'] = tf.expand_dims(
                        detection_masks_reframed, 0)
                image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

                # Run inference
                output_dict = sess.run(tensor_dict,
                                       feed_dict={image_tensor: np.expand_dims(image, 0)})

                # all outputs are float32 numpy arrays, so convert types as appropriate
                output_dict['num_detections'] = int(output_dict['num_detections'][0])
                output_dict['detection_classes'] = output_dict[
                    'detection_classes'][0].astype(np.uint8)
                output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
                output_dict['detection_scores'] = output_dict['detection_scores'][0]
                if 'detection_masks' in output_dict:
                    output_dict['detection_masks'] = output_dict['detection_masks'][0]

            # output_dict["detection_boxes"], '     '
            # , output_dict["detection_scores"]
            # for i in output_dict.__len__():
            #     print(output_dict["detection_scores"][i])
            # output_dict["detection_boxes"]

        end = time.time() - start
        print("cost:", end, 's')

    return output_dict


import os

label = ["open_eyes", "close_eyes", "phone", "smoke", "yawn", "side_face", "face"]
count = 0
# cap = cv2.VideoCapture(0)
for image in sorted(TEST_IMAGE_PATHS):
    # cv2.waitKey(1)
    # ret, frame = cap.read()
    # image = tf.image.resize_images(image,(300,300,3))
    # print(os.path.basename(image))
    str1 = os.path.basename(image)

    image = Image.open(image)
    image_np = load_image_into_numpy_array(image)
    # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
    # image_np_expanded = np.expand_dims(image_np, axis=0)
    # Actual detection.
    output_dict = run_inference_for_single_image(image_np, detection_graph)
    # Visualization of the results of a detection.
    # print(list(output_dict['detection_scores']))
    im_width, im_height = image.size

    listString = "<annotation>\n" + \
                 "<folder>dabai</folder>\n" + \
                 "<filename>" + str1 + "</filename>\n" + \
                 "<path>\n" + \
                 "C:\\dabai\\" + str1 + "\n" + \
                 "</path>\n" + \
                 "<source>\n" + \
                 "<database>Unknown</database>\n" + \
                 "</source>\n" + \
                 "<size>\n" + \
                 "<width>" + im_width + "</width>\n" + \
                 "<height>" + im_height + "</height>\n" + \
                 "<depth>3</depth>\n" + \
                 "</size>\n" + \
                 "<segmented>0</segmented>\n"
    isHave = False
    for indx, asdf in enumerate(output_dict["detection_scores"]):
        if (asdf > JILV):
            isHave = True
            ymin, xmin, ymax, xmax = tuple(output_dict["detection_boxes"][indx].tolist())
            left, right, top, bottom = (xmin * im_width, xmax * im_width,
                                        ymin * im_height, ymax * im_height)
            # print(xmin * im_width, xmax * im_width,
            #                       ymin * im_height, ymax * im_height)
            listString = listString + "<object>\n" \
                                      "<name>" + label[output_dict["detection_classes"][indx] - 1] + "</name>\n" \
                                                                                                     "<pose>Unspecified</pose>\n" \
                                                                                                     "<truncated>0</truncated>\n" \
                                                                                                     "<difficult>0</difficult>\n" \
                                                                                                     "<bndbox>\n" \
                                                                                                     "<xmin>%d</xmin>\n" \
                                                                                                     "<ymin>%d</ymin>\n" \
                                                                                                     "<xmax>%d</xmax>\n" \
                                                                                                     "<ymax>%d</ymax>\n" \
                                                                                                     "</bndbox>\n" \
                                                                                                     "</object>\n" % (
                             left, top, right, bottom,)
    listString = listString+"</annotation>"
    if (isHave != True):
        continue
    print(count)
    fp = open("./test_images/" + str1.replace(".jpg", "") + '.xml', "w")
    fp.write(listString)
    fp.close()
    count += 1

# cap.release()
# cv2.destroyAllWindows()
