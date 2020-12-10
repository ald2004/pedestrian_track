import cv2
import os
import json
import time
import uuid
import numpy as np
from multiprocessing import Process
from api.CameraApi import (
    get_ins,
    start
)
import ffmpeg
from api.convertimg import gen_camera_frame
import fire
from api.yolo_reid_pint_api import YoloReidPintAPIAlgorithm as pint_api
from api.yolo_reid_pint_api import Config
from api.yolo_reid_pint_api import cosine_distance
from api.plot_utils import plot_multi_images
from api.timer import Timer


# class Config():
#     def __init__(self, json_path=None):
#         self.config = {}
#         with open(json_path, "r", encoding='utf-8') as f:
#             self.config = json.loads(f.read())
#         self.config['FIFO'] = '/dev/shm/img_q'
#         self.config['w'] = 640
#         self.config['h'] = 480
#         self.config['c'] = 3


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

    def pint_main(self, ffmpegprocess):
        # imgs_list = ['./images/boe_test.jpg', ]
        # print(config)

        # imgs = []
        # for img_path in imgs_list:
        #     img = cv2.imread(img_path)
        #     imgs.append(img)

        # get config
        # model = pint_api(self.config)
        # get frame
        while not os.path.exists(self.config['FIFO']):
            print(f'wait for frame pipe : {self.config["FIFO"]}')
            time.sleep(1)
        # for _ in range(10):
        # while 1:
        xor = 0
        outfifo=open('/dev/shm/rtmp_q','wb')
        with open(self.config['FIFO'], 'rb') as fifo:
        # cap = cv2.VideoCapture(0)
            while 1:
                data = fifo.read(self.config['w'] * self.config['h'] * self.config['c'])
                # print(f'len of data is :{len(data)}')
                # xor += 1
                # if xor != 3:
                #     continue
                # xor = 0
                # f, img = cap.read()
                if len(data):
                # if f:
                    buff = np.frombuffer(data, np.uint8)
                    imgx = cv2.imdecode(buff, cv2.IMREAD_COLOR)
                    # img = cv2.resize(imgx, (320,240))
                    # features, boxes, detect_result = model.run([img])
                    #
                    # if detect_result:
                    #     plot_multi_images([img], detect_result)
                    # else :
                    #     _, _, detect_result = model.run([img])
                    #     plot_multi_images([img], detect_result)
                    ret2, frame2 = cv2.imencode('.png', imgx)
                    outfifo.write(frame2)
                    # ffmpegprocess.stdin.write(frame2.tobytes())
                    # print("#" * 40)
                    # print(features)
                    # print("#" * 40)
                    # print(boxes)
                    # print("#" * 40)
                    # print(detect_result)
                    # print("#" * 40)
                    # dist = cosine_distance(features[0][0], features[0][1])
                    # print(len(features[0]), len(features[0][0]))
                    # print(dist)

                    # cv2.imwrite(f'/dev/shm/detection_result_{uuid.uuid4().hex}.jpg', img)

                    # print("the merge vision run time is {:.2f} ms".format((end - start) * 1000))
                # else:
                #     print('data is none')


def get_camera_frame(_):
    print('=' * 60)
    print(f'start get camera frame with process :{Process.name}')
    print('=' * 60)
    spam = get_ins()
    start(spam)


def pintmain(process):
    xx = demo_classer()
    xx.pint_main(process)


if __name__ == "__main__":
    # process = (
    #     ffmpeg
    #         .input('pipe:', r='6', hwaccel='vdpau', hwaccel_device='/dev/dri/card0')
    #         .output('rtmp://192.168.8.121/live/bbb', vcodec='libx264', pix_fmt='yuv420p', preset='veryfast',
    #                 r='20', g='50', video_bitrate='1.4M', maxrate='2M', bufsize='2M', segment_time='6',
    #                 format='flv',
    #                 #**{'c:v': 'h264_rkmpp'}
    #                 )
    #         .run_async(pipe_stdin=True))
    gcf = Process(target=get_camera_frame, args=(None,))
    pm = Process(target=pintmain, args=(None,))
    # gci = Process(target=gen_camera_frame,args=(None,))
    gcf.start()
    pm.start()
    # fire.Fire(demo_classer)
    # gcf.join()
    # pm.join()
    while 1:
        time.sleep(30)
