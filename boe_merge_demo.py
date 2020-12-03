import cv2
import os
import json
import time
import uuid
import numpy as np


from api.yolo_reid_pint_api import YoloReidPintAPIAlgorithm as pint_api
from api.yolo_reid_pint_api import Config
from api.yolo_reid_pint_api import cosine_distance
from api.plot_utils import plot_multi_images
from api.timer import Timer


class Config():
    def __init__(self, json_path=None):
        self.config = {}
        with open(json_path, "r", encoding='utf-8') as f:
            self.config = json.loads(f.read())


class demo_classer():
    def __init__(self):
        with open("./config/yolo_reid_config.json", "r", encoding='utf-8') as f:
            self.config = json.loads(f.read())
        self.model = pint_api(self.config)
        self.timer = Timer()


# test only one image
def pint_main():
    imgs_list = ['./images/boe_test.jpg', ]
    config = Config(json_path="./config/yolo_reid_config.json").config
    # print(config)

    imgs = []
    for img_path in imgs_list:
        img = cv2.imread(img_path)
        imgs.append(img)

    model = pint_api(config)

    start = time.time()
    features, boxes, detect_result = model.run(imgs)
    end = time.time()
    print("#" * 40)
    print("the merge vision run time is {:.2f} ms".format((end - start) * 1000))

    dist = cosine_distance(features[0][0], features[0][1])
    print(len(features[0]), len(features[0][0]))
    print(dist)

    plot_multi_images(imgs, detect_result)

    for i, img in enumerate(imgs):
        # cv2.imshow('Detection result', imgs[i])
        cv2.imwrite(f'detection_result_{uuid.uuid4().hex}.jpg', imgs[i])
        # cv2.waitKey(0)


def pp_main():
    dtr = demo_classer()



if __name__ == "__main__":
    pp_main()
    # pint_main()
