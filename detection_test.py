import numpy as np
import tensorflow as tf
import os
import glob
import cv2
import time

PATH_TO_CKPT = 'frozen_inference_graph_truc_nohardexample_1028.pb'
image_path = 'test_images'
JILV = 0.6

PATH_TO_LABELS = {1: "open_eyes", 2: "close_eyes", 3: "phone", 4: "smoke", 5: "yawn", 6: "side_face", 7: "face"}

images = glob.glob(image_path + "/*.jpg")
count = 0
NUM_CLASSES = 7
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def load_graph(model_path):
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(model_path, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')

    return detection_graph


detection_graph = load_graph(PATH_TO_CKPT)
sess = tf.Session(graph=detection_graph)

for image in images:
    str1 = os.path.basename(image)
    img = cv2.imread(image)
    # im_width, im_height = img.size

    im_width = img.shape[1]
    im_height = img.shape[0]
    ops = detection_graph.get_operations()
    all_tensor_names = {output.name for op in ops for output in op.outputs}
    tensor_dict = {}
    for key in [
        'num_detections', 'detection_boxes', 'detection_scores',
        'detection_classes', 'detection_masks'
    ]:
        tensor_name = key + ':0'
        if tensor_name in all_tensor_names:
            tensor_dict[key] = detection_graph.get_tensor_by_name(
                tensor_name)
    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

    output_dict = sess.run(tensor_dict,
                           feed_dict={image_tensor: np.expand_dims(img, 0)})

    output_dict['num_detections'] = int(output_dict['num_detections'][0])
    output_dict['detection_classes'] = output_dict[
        'detection_classes'][0].astype(np.uint8)
    output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
    output_dict['detection_scores'] = output_dict['detection_scores'][0]

    listString = "<annotation>\n" + \
                 "<folder>xiucai</folder>\n" + \
                 "<filename>" + str1 + "</filename>\n" + \
                 "<path>\n" + \
                 "C:\\xiucai\\" + str1 + "\n" + \
                 "</path>\n" + \
                 "<source>\n" + \
                 "<database>Unknown</database>\n" + \
                 "</source>\n" + \
                 "<size>\n" + \
                 "<width>640</width>\n" + \
                 "<height>480</height>\n" + \
                 "<depth>3</depth>\n" + \
                 "</size>\n" + \
                 "<segmented>0</segmented>\n"
    isHave = False
    for indx, asdf in enumerate(output_dict["detection_scores"]):
        if (asdf > JILV):
            isHave = True
            # print(output_dict["detection_boxes"][indx])
            # if (output_dict["detection_classes"][indx] == 7):
            # print(label[output_dict["detection_classes"][indx] - 1], asdf)

            ymin, xmin, ymax, xmax = tuple(output_dict["detection_boxes"][indx].tolist())
            left, right, top, bottom = (xmin * im_width, xmax * im_width,
                                        ymin * im_height, ymax * im_height)
            # print(xmin * im_width, xmax * im_width,
            #                       ymin * im_height, ymax * im_height)
            listString = listString + "<object>\n" \
                                      "<name>" + PATH_TO_LABELS[output_dict["detection_classes"][indx]] + "</name>\n" \
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
    listString = listString + "</annotation>"
    print(count)
    if (isHave != True):
        continue
    fp = open("./test_images/" + str1.replace(".jpg", "") + '.xml', "w")
    fp.write(listString)
    fp.close()
    count += 1
    # for indx,boxes in enumerate(output_dict["detection_boxes"]):
    #     ymin, xmin, ymax, xmax = np.split(boxes,indices_or_sections=4)
    #     (left, right, top, bottom) = (xmin * width, xmax * width,
    #                                   ymin * heigth, ymax * heigth)
    #     if int(100*output_dict["detection_scores"][indx])>0.8:
    #         if output_dict["detection_classes"][indx]==7:
    #             cv2.rectangle(img, (left, top), (right, bottom), (0, 255, 0), 2)
    #         if output_dict["detection_classes"][indx]==1:
    #             cv2.rectangle(img, (left, top), (right, bottom), (255, 0, 0), 2)
    #         if output_dict["detection_classes"][indx]==2:
    #             cv2.rectangle(img, (left, top), (right, bottom), (0, 0, 255), 2)
    #         y = top - 15 if top - 15 > 15 else top + 15
    #         label = output_dict["detection_classes"][indx]
    #         cv2.putText(img, PATH_TO_LABELS[label], (left, y), cv2.FONT_HERSHEY_SIMPLEX,
    #                     0.75, (0, 255, 0), 2)
    # cv2.imshow("Image", img)
    # cv2.waitKey(1000)

sess.close()
