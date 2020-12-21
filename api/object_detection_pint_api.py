import os
from api import nms, yolo_detection

import cv2
import numpy as np
import json
import time

try:
    from pint_mini import pint_mini
except:
    import pint_mini


class ObjectDectionPintAPIAlgorithm():

    def __init__(self, config):
        self.config = config
        with open(os.path.normpath('/opt/pedestrian_track/config/coco_names.json'), "r") as f:
            self.class_names = json.load(f)
        self.anno_id = 0

        self._build()

    def _build(self):
        in_shape = [self.config['batch'], self.config['height'], self.config['width'], 3]
        in_names = pint_mini.mini.StrVector(["input_data"])
        out_names = pint_mini.mini.StrVector(["yolov3/yolov3_head/Conv_6/BiasAdd",
                                              "yolov3/yolov3_head/Conv_14/BiasAdd",
                                              "yolov3/yolov3_head/Conv_22/BiasAdd"])

        options = pint_mini.mini.Options()

        if self.config["tuning"]:
            options.fusion_on = self.config["fusion_on"]
            options.pitch_on = self.config["pitch_on"]
            options.prt_mode = self.config["prt_mode"]

        if self.config['quantization_enable']:
            options.json_path = self.config["tuning_i8"]
            graph = pint_mini.buildGraph(self.config['model_path_pmd'], 3)
            self.sess = pint_mini.createSessionFromPmd(graph, options=options)
        else:
            options.json_path = self.config["tuning_fp32"]
            print(self.config['model_path'])
            graph = pint_mini.buildGraph(self.config['model_path'], 0, in_names=in_names, out_names=out_names)
            self.sess = pint_mini.createSession(graph, [in_shape], options)

        if self.config['post_on_host']:
            self.yolo_det = yolo_detection.YOLO_Predict(img_size=[self.config['height'], self.config['width']])
        else:
            # to do 
            pass

    def _pre_process(self, imgs):

        def read_image(input_image, height, width):
            img = cv2.resize(input_image, (height, width), interpolation=0)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = np.asarray(img, np.float32)
            img = img / 255.

            return img

        tmp = []
        for img in imgs:
            img_ = read_image(img, self.config['height'], self.config['width'])
            tmp.append(img_)

        return np.asarray(tmp)

    def _post_process(self, det):
        boxes_, scores_, confs_ = self.yolo_det.predict(tuple(det))
        # print(boxes_.shape, scores_.shape, confs_.shape)
        pred_scores = scores_ * confs_

        ret = []
        boxes_ret = []
        scores_ret = []
        labels_ret = []
        for i in range(det[0].shape[0]):
            boxes = boxes_[i, :, :]
            scores = pred_scores[i, :, :]

            boxes, scores, labels = nms.cpu_nms(boxes,
                                                scores,
                                                self.config["class_number"],
                                                max_boxes=self.config['max_num'],
                                                score_thresh=self.config['score_threshold'],
                                                iou_thresh=self.config['iou_threshold'])
            # print(boxes, scores, labels)
            if boxes is None:
                ret.append([])
                continue

            boxes[:, [0, 2]] = boxes[:, [0, 2]] / float(self.config['width'])
            boxes[:, [1, 3]] = boxes[:, [1, 3]] / float(self.config['height'])

            one_image_ret = []
            boxes_ret.append(boxes)
            scores_ret.append(scores)
            labels_ret.append(labels)
            for j, box in enumerate(boxes):
                element = dict()
                element['box'] = box
                element['class_name'] = self.class_names[str(labels[j])]
                if element['class_name'] != 'person':
                    continue
                element['score'] = scores[j]
                element['class_id'] = labels[j]
                one_image_ret.append(element)

            ret.append(one_image_ret)

        return ret, boxes_ret, scores_ret, labels_ret

    def run(self, f):
        imgs = self._pre_process(f)

        # start = time.time()
        result = self.sess.predict([imgs])
        # print(result)
        # end = time.time()
        # print("#" * 40)
        # print("object detection run time is {:.2f} ms".format((end-start)*1000))

        ret = self._post_process(result)

        return ret


model = ObjectDectionPintAPIAlgorithm
