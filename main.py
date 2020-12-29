import cv2
import glob
from api.yolo_reid_pint_api import YoloReidPintAPIAlgorithm as pint_api
import json
from api.timer import Timer


class demo_classer(object):
    def __init__(self):
        with open("./config/yolo_reid_config.json", "r", encoding='utf-8') as f:
            self.config = json.loads(f.read())
        self.model = pint_api(self.config)
        self.timer = Timer()
        self.config['FIFO'] = '/dev/shm/img_q'
        self.config['w'] = 640
        self.config['h'] = 480
        self.config['c'] = 3
        self.config['RTMP_PATH'] = 'rtmp://192.168.8.121/live/bbb'

    def pint_main(self):
        videos = glob.glob("./videos/*.avi")
        for video in videos:
            cap = cv2.VideoCapture(video)
            cnt = 0
            while cap.isOpened():
                cnt += 1
                if cnt > 100:
                    break
                f, img = cap.retrieve()
                features, boxes, detect_result = self.model.run([img])
                print(features[0])


xx = demo_classer()
xx.pint_main()
