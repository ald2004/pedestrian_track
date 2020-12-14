import json
import numpy as np
import copy

from api.reid_pint_api import FeatureExtractionPintAPIAlgorithm as reid_pint_api
from api.object_detection_pint_api import ObjectDectionPintAPIAlgorithm as yolo_pint_api
from api.plot_utils import plot_multi_images


class Config():
    def __init__(self, json_path=None):
        self.config = {}
        with open(json_path, "r", encoding='utf-8') as f:
            self.config = json.loads(f.read())


def cosine_distance(a, b, data_is_normalized=False):
    if not data_is_normalized:
        a = np.asarray(a) / np.linalg.norm(a, axis=0, keepdims=True)
        b = np.asarray(b) / np.linalg.norm(b, axis=0, keepdims=True)
    return 1. - np.dot(a, b.T)


class YoloReidPintAPIAlgorithm():
    def __init__(self, config):
        self.config = config
        self.yolo_config = Config(json_path=self.config["yolo_config_path"]).config
        self.reid_config = Config(json_path=self.config["reid_config_path"]).config
        self.detector = yolo_pint_api(self.yolo_config)
        self.feature_extracter = reid_pint_api(self.reid_config)

    def pre_process(self, imgs):
        pass

    def post_process(self, det):
        pass

    def run(self, images):
        _images = copy.deepcopy(images)
        det_result = self.detector.run(images)
        # boxes = [[[292, 152, 63, 177], [453, 208, 27, 67]], ]
        boxes = self.convert_boxes(_images, det_result[0])
        features = self.feature_extracter.run(images, boxes)
        return features, boxes, det_result[0]
        # return None, None, det_result[0]

    def convert_boxes(self, images, detections):
        boxes = []
        for i, img in enumerate(images):
            h, w, c = img.shape
            sigle_image_boxes = []
            objects = detections[i]

            temp_box = []
            for object in objects:
                temp_box = copy.deepcopy(object["box"])
                temp_box[2:] = temp_box[2:] - temp_box[:2]
                temp_box = temp_box.tolist()
                temp_box[0] = temp_box[0] * w
                temp_box[1] = temp_box[1] * h
                temp_box[2] = temp_box[2] * w
                temp_box[3] = temp_box[3] * h
                temp_box = [int(x) for x in temp_box]
                sigle_image_boxes.append(temp_box)
            boxes.append(sigle_image_boxes)

        return boxes


model = YoloReidPintAPIAlgorithm
