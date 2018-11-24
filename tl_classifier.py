from styx_msgs.msg import TrafficLight

import numpy as np
import cv2
import tensorflow as tf
import os
from glob import glob
from PIL import Image
from io import StringIO

cwd = os.path.dirname(os.path.realpath(__file__))


class TLClassifier(object):
    def __init__(self):
        #TODO load classifier

        # For traffic light detection : Kinji Sato
        os.chdir(cwd)
        SSD_GRAPH_FILE = 'ssd_mobilenet_v1_coco_2018_01_28/frozen_inference_graph.pb'

        #self.detection_graph = self.load_graph(SSD_GRAPH_FILE)


        self.detection_graph = tf.Graph()
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.tl_box = None

        with self.detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(SSD_GRAPH_FILE, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')

            self.sess = tf.Session(graph=self.detection_graph, config=config)
            # The input placeholder for the image.
            # `get_tensor_by_name` returns the Tensor with the associated name in the Graph
            self.image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')
            # Each box represents a part of the image where a particular object was detected.
            self.detection_boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')
            # Each score represent how level of confidence for each of the objects.
            # Score is shown on the result image, together with the class label.
            self.detection_scores =self.detection_graph.get_tensor_by_name('detection_scores:0')
            self.detection_classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
            self.num_detections =self.detection_graph.get_tensor_by_name('num_detections:0')


    def get_classification(self, image):
        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        #TODO implement light color prediction
        return TrafficLight.UNKNOWN

    def load_graph(self, graph_file):
        graph = tf.Graph()
        with graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(graph_file, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')
        return graph


    def filter_boxes(min_score, boxes, scores, classes):
        """Return boxes with a confidence >= `min_score`"""
        n = len(classes)
        idxs = []
        for i in range(n):
            if scores[i] >= min_score:
                idxs.append(i)

        filtered_boxes = boxes[idxs, ...]
        filtered_scores = scores[idxs, ...]
        filtered_classes = classes[idxs, ...]
        return filtered_boxes, filtered_scores, filtered_classes

    def get_localization(self, image):
        #image_expanded = np.expand_dims(image, axis=0)
        #image_resized = image.resize((128, 96))
        image_resized = image
        image_np = np.expand_dims(np.asarray(image_resized, dtype=np.uint8), 0)

        with self.detection_graph.as_default():
            (boxes, scores, classes, num_detections) = self.sess.run(
                [self.detection_boxes, self.detection_scores, self.detection_classes, self.num_detections],
                feed_dict={self.image_tensor: image_np})

            # Remove unnecessary dimensions
            boxes = np.squeeze(boxes)
            scores = np.squeeze(scores)
            classes = np.squeeze(classes)
            # detections = np.squeeze(detections)
            cls = classes.tolist() #array to list

            confidence_cutoff = 0.5  # 0.9
            # Filter boxes with a confidence score less than `confidence_cutoff`
            boxes, scores, classes = filter_boxes(confidence_cutoff, boxes, scores, classes)

            # Find the first occurence of traffic light detection id=10(=traffic light)
            idx = next((i for i, v in enumerate(cls) if v == 10.), None)
            # If there is no detection
            if idx == None:
                box=[0, 0, 0, 0]
                print('no detection!')
            else:
                dim = image.shape[0:2]
                box = self.box_normal_to_pixel(boxes[idx], dim)
                box_h = box[2] - box[0]
                box_w = box[3] - box[1]
                ratio = box_h/(box_w + 0.01)
                # if the box is too small, N pixels
                if (box_h <20) or (box_w<20):
                    box =[0, 0, 0, 0]
                    print('box too small!', box_h, box_w)
                # if the h-w ratio is not right, 1.5 for simulator, 0.5 for site
                elif (ratio < 1.5):
                    box =[0, 0, 0, 0]
                    print('wrong h-w ratio', ratio)
                else:
                    print(box)
                    print('localization confidence: ', scores[idx])

            self.tl_box = box
        return box
