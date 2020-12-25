import cv2
import os

import json
import time
import uuid
import numpy as np
from itertools import cycle
import sqlite3
from multiprocessing import Process
# from api.CameraApi import (
#     get_ins,
#     start
# )
# from api.cppencoder import (
#     cur,
#     inData
# )
import copy
import multiprocessing
import threading
from multiprocessing import Process, Queue
import ffmpeg
from api.convertimg import gen_camera_frame
import fire
from api.yolo_reid_pint_api import YoloReidPintAPIAlgorithm as pint_api
from api.yolo_reid_pint_api import Config
from api.yolo_reid_pint_api import cosine_distance
from sklearn.metrics.pairwise import cosine_similarity
from api.plot_utils import plot_multi_images
from api.timer import Timer
import sqlite3
import asyncio
import websockets
import datetime
import glob
import ctypes
from ctypes import *
import threading
import numpy as np

import datetime
import os
import cv2
import copy

IMAGE_PIPE = '/dev/shm/img_q'
RTMP_PIPE = '/dev/shm/rtmp_q'
w = 640
h = 480
yoloW = 416,
c = 3
THRES = .3
THRES_ALL = .7
HOST = "0.0.0.0"
# np.set_printoptions(threshold=9223372036854775807)

connected = set()
total_count = 0
q0_count, q1_count, q2_count, q3_count, \
q4_count, q5_count = 0, 0, 0, 0, 0, 0
h0, h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, \
h13, h14, h15, h16, h17, h18, h19, h20, h21, h22, h23 = \
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
realHeat = []
realTrack = {
    'id': [],
    'point': [],
}


# class Config():
#     def __init__(self, json_path=None):
#         self.config = {}
#         with open(json_path, "r", encoding='utf-8') as f:
#             self.config = json.loads(f.read())
#         self.config['FIFO'] = '/dev/shm/img_q'
#         self.config['w'] = 640
#         self.config['h'] = 480
#         self.config['c'] = 3

def distinctlist(intlist):
    s = set()
    r = []
    for i in intlist:
        s.add((i[0], i[1], i[2], i[3]))
    for i in s:
        r.append([i[0], i[1], i[2], i[3]])
    return r


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
        self.conn = sqlite3.connect('aquarium.db')
        self.c = self.conn.cursor()

    def result_to_db(self, f, b, d):
        if not len(b):
            return
        # f (128,)
        # b [[183, 163, 484, 317], [52, 121, 38, 48], [13, 124, 46, 70], [12, 172, 41, 40], [401, 172, 46, 84], [224, 280, 202, 193]]
        # d
        # [
        #     {'box': array([0.28747933, 0.34066093, 1.04517841, 1.00288587]), 'class_name': 'person',
        #      'score': 0.99528664, 'class_id': 0},
        #     {'box': array([0.0814497, 0.25219855, 0.14155266, 0.35405122]), 'class_name': 'person', 'score': 0.9562466,
        #      'class_id': 0},
        #     {'box': array([0.02163462, 0.2603192, 0.09375, 0.40695381]), 'class_name': 'person', 'score': 0.78090733,
        #      'class_id': 0},
        #     {'box': array([0.01887899, 0.35872319, 0.08315149, 0.44294513]), 'class_name': 'person',
        #      'score': 0.64942235, 'class_id': 0},
        #     {'box': array([0.62744258, 0.35946582, 0.69955796, 0.53540744]), 'class_name': 'person',
        #      'score': 0.36657786, 'class_id': 0},
        #     {'box': array([0.35106538, 0.58400088, 0.66795829, 0.98621112]), 'class_name': 'chair', 'score': 0.62029713,
        #      'class_id': 56}
        # ]
        # filename = f"/dev/shm/{uuid.uuid4().hex}"
        filename = f"/dev/shm/{time.time()}"
        np.savez_compressed(filename, allow_pickle=True, f=f, b=b, d=d)
        # with open('/dev/shm/result_tmp_txt','w') as fid:
        #     fid.writelines([str(a) for a in f])
        #     fid.write("="*100)
        #     fid.writelines([str(a) for a in b])
        #     fid.write("=" * 100)
        #     fid.writelines([str(a) for a in d])

    def pint_main(self, q: Queue, outq: Queue, detstatisq: Queue):
        # imgs_list = ['./images/boe_test.jpg', ]
        # print(config)

        # imgs = []
        # for img_path in imgs_list:
        #     img = cv2.imread(img_path)
        #     imgs.append(img)

        # get config
        # model = pint_api(self.config)
        # get frame
        # while not os.path.exists(self.config['FIFO']):
        #     print(f'wait for frame pipe : {self.config["FIFO"]}')
        #     time.sleep(1)
        # # for _ in range(10):
        # # while 1:
        xor = 0
        # outfifo = open('/dev/shm/rtmp_q', 'wb')
        # with open(self.config['FIFO'], 'rb') as fifo:
        # cap = cv2.VideoCapture(0)
        while 1:
            # data = fifo.read(self.config['w'] * self.config['h'] * self.config['c'])
            # print(f'len of data is :{len(data)}')
            # xor += 1
            # if xor != 3:
            #     continue
            # xor = 0
            # f, img = cap.read()
            try:
                img: np.ndarray = q.get()
                features, boxes, detect_result = self.model.run([img])
            except:
                time.sleep(.2)
                continue
            try:
                if len(features[0]):
                    detstatisq.put((features, boxes, detect_result))
            except:
                print(f"detstatisq err {features, boxes, detect_result}")
            # xor += 1
            # if not (xor - 3):
            #     # self.result_to_db(features, boxes, detect_result)
            #     xor = 0
            # plot_multi_images([img], detect_result)

            # ret2, frame2 = cv2.imencode('.png', img)
            # outq.put(img)
            outq.put(detect_result)
            # if len(data):
            # if f:
            # buff = np.frombuffer(data, np.uint8)
            # imgx = cv2.imdecode(buff, cv2.IMREAD_COLOR)
            # img = cv2.resize(imgx, (320,240))
            # features, boxes, detect_result = model.run([img])
            #
            # if detect_result:
            #     plot_multi_images([img], detect_result)
            # else :
            #     _, _, detect_result = model.run([img])
            #     plot_multi_images([img], detect_result)
            # ret2, frame2 = cv2.imencode('.png', imgx)
            # outfifo.write(frame2)
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

    def pint_main_detection(self, imq, boxq):
        while 1:
            try:
                img: np.ndarray = imq.get()
                boxes, detect_result = self.model.run_detect([img])
            except:
                time.sleep(.2)
                continue
            try:
                if len(boxes):
                    boxq.put(detect_result)
            except:
                print(f"detstatisq err {boxes, detect_result}")
            # xor += 1
            # if not (xor - 3):
            #     # self.result_to_db(features, boxes, detect_result)
            #     xor = 0
            # plot_multi_images([img], detect_result)

            # ret2, frame2 = cv2.imencode('.png', img)
            # outq.put(img)
            # outq.put(detect_result)


def get_camera_frame(_):
    # print('=' * 60)
    # print(f'start get camera frame with process :{Process.name}')
    # print('=' * 60)
    try:
        spam = get_ins()
        start(spam)
    except:
        pass


def pintmain(q, outq, detstatisq):
    xx = demo_classer()
    xx.pint_main(q, outq, detstatisq)


def pintmain_detection(imq, boxq):
    xx = demo_classer()
    xx.pint_main_detection(imq, boxq)


def p_getframe_python_to_queue(q: Queue):
    with open(IMAGE_PIPE, 'rb') as fifo:
        print("FIFO opened")
        while True:
            # image_stream = io.BytesIO()
            data = fifo.read(w * h * c)
            if not len(data):
                continue

            buff = np.frombuffer(data, np.uint8)
            imgx = cv2.imdecode(buff, cv2.IMREAD_COLOR)
            try:
                q.put(imgx)
            except:
                print('q full')
                try:
                    for _ in range(20):
                        q.get_nowait()
                except:
                    pass
    # cap = cv2.VideoCapture(0)
    # while 1:
    #     f, i = cap.read()
    #     if f:
    #         try:
    #             q.put(i)
    #         except:
    #             print('q full')
    #             try:
    #                 for _ in range(200):
    #                     q.get_nowait()
    #             except:
    #                 pass


def put_bgr_encode_put_to_ffmpeg(frame, cur, inData):
    # print('qqqqqqqqqqqqqqq')
    # print(frame)
    # print(frame.shape)
    yuv = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV_I420)
    # print(yuv.shape)
    # print('rrrrrrrrrrrrrrrr')
    frame_for_encoder = copy.deepcopy(yuv)
    # print(frame_for_encoder.shape)
    # print(frame_for_encoder)
    # print('wwwwwwwwwwwwww')
    inData.buf = (ctypes.c_ubyte * 460800).from_buffer_copy(frame_for_encoder)
    # print(inData.buf)
    # print('eeeeeeeeeeeeeeeee')
    cur.inputBgr(inData)
    # print('etttttttttttttttttttttt')
    img = cur.readH264()
    # print(img.len)
    # print('yyyyyyyyyyyyyyyy')
    ffmpegprocess.stdin.write(bytes(img.buf1[:img.len]))
    # print('zzzzzzzzzzzz')
    # try:
    #     ffmpegprocess.stdin.write(bytes(img.buf1[:img.len]))
    # except:
    #     pass


def p_putframe_python_to_pipe(q: Queue, boxq: Queue):
    h264hearder = None
    with open('/opt/pedestrian_track/api/output_short_header_25fps.h264', 'rb') as fid:
        # h264hearder = fid.read(383)
        h264hearder = fid.read()
    ffmpegprocess.stdin.write(h264hearder)
    # while not os.path.exists(RTMP_PIPE):
    #     print(f'wait for frame pipe : {RTMP_PIPE}')
    #     time.sleep(1)
    cur = CDLL(os.path.join(os.environ.get('ESW_PATH', './'), "libEncode_no_std_print.so"), RTLD_GLOBAL)

    class dataInPack(Structure):
        _fields_ = [("buf", c_ubyte * 460800)]  # buf1

    class dataOutPack(Structure):
        _fields_ = [("len", c_int),  # width
                    ("buf1", c_ubyte * 2000000)]  # buf1

    cur.readH264.restype = (dataOutPack)
    inData = dataInPack()
    cur.create_encoder(640, 480, 25)

    # i = q.get()
    # yuv = cv2.cvtColor(i, cv2.COLOR_BGR2YUV_I420)
    # frame_for_encoder = copy.deepcopy(yuv)
    # inData.buf = (ctypes.c_ubyte * 460800).from_buffer_copy(frame_for_encoder)
    # cur.inputBgr(inData)
    # img = cur.readH264()
    # print(img.len)
    # print('yyyyyyyyyyyyyyyy')
    # # ffmpegprocess.stdin.write(bytes(img.buf1[:img.len]))

    def put_real_stream(outfifo, outq, q, cur, inData):
        while 1:
            try:
                i = q.get()
                # ret2, frame2 = cv2.imencode('.png', i)
                # outfifo.write(frame2)
                # print('zzzzzzzzzzzz')
                put_bgr_encode_put_to_ffmpeg(i, cur, inData)
            except:
                time.sleep(.5)
                print(f'none q img')
                # raise

    def put_det_stream(outfifo, outq: multiprocessing.Queue, realq, cur, inData):
        detect_result = outq.get()
        # print('aaaaaaaa')
        while 1:
            # outi = outq.get()
            # print('ddddddddddd')
            try:
                current_detect_result = outq.get_nowait()
                detect_result = current_detect_result
            except:
                # print('ssssssssssss')
                current_detect_result = detect_result
            try:
                # print('gggggggggggggg')
                img = realq.get()
                # print('ssssgggggggggggggggg')
                if img.shape[2] == 3:
                    # plot_multi_images([img], detect_result)
                    plot_multi_images([img], current_detect_result, line_thickness=2)
                    # ret2, frame3 = cv2.imencode('.png', img)
                    # outfifo.write(frame3)
                    # print('hhhhhhhhhhhhh')
                    put_bgr_encode_put_to_ffmpeg(img, cur, inData)
            except:
                pass
            # outfifo.write(img)

    # with open(RTMP_PIPE, 'wb') as outfifo:
    # t1 = threading.Thread(target=put_real_stream, args=(None, outq, q, cur, inData))
    # t2 = threading.Thread(target=put_det_stream, args=(None, boxq, q, cur, inData))
    t1 = threading.Thread(target=put_det_stream, args=(None, boxq, q, cur, inData))
    # t1 = multiprocessing.Process(target=put_real_stream, args=(outfifo, q))
    # t2 = multiprocessing.Process(target=put_det_stream, args=(outfifo, outq, q))

    # print('xxxxxxxxxxxxxxx')
    t1.start()
    # t2.start()
    while 1:
        time.sleep(99)
        # while 1:
        #     outi = outq.get()
        #     ret2, frame3 = cv2.imencode('.png', outi)
        #     outfifo.write(frame3)
        # try:
        #     outi=outq.get()
        #     ret2, frame3 = cv2.imencode('.png', outi)
        #     outfifo.write(frame3)
        # except:
        #     pass
        # try:
        #     i = q.get()
        #     ret2, frame2 = cv2.imencode('.png', i)
        #     outfifo.write(frame2)
        # except:
        #     time.sleep(.5)
        #     print(f'none q img')


async def server(websocket, path: str):
    if len(connected) > 50:
        return
    global total_count, q0_count, q1_count, q2_count, q3_count, \
        q4_count, q5_count, h0, h1, h2, h3, h4, h5, \
        h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, \
        h16, h17, h18, h19, h20, h21, h22, h23, realHeat, realTrack

    q0_count, q1_count, q2_count, q3_count, \
    q4_count, q5_count = 10, 20, 30, 40, 50, 60
    if path.endswith('realTrack'):
        connected.add(websocket)
    # pk = path.split('/')[-1]
    # connected[pk] = websocket
    # realHeat
    # realTrack
    # realHeat
    # reportDetail
    # reportByHour
    # print(pk,'111111111111')
    try:
        # async for message in websocket:
        # for conn in connected:
        # print(f'pth is :{path}')
        if path.endswith('reportDetail'):
            # print(path, '111111111111')
            # xx = {
            #     "code": 200,
            #     "data": [
            #         {"areaCount": {"A": 3938, "B": 0, "C": 11, "D": 0},
            #          "ageCount": {"46-60": "10%", "61": "15%", "31-45": "5%", "21-30": "50%", "0-20": "20%"},
            #          "genderCount": {"men": {"num": 2, "percent": "10%"}, "women": {"num": 4, "percent": "20%"}},
            #          "totalCount": 31, "faceCount": 0,
            #          "personCount": {"60-120": "0", "0-15": "18", "30-45": "0", "15-30": "0", ">120": "0", "45-60": "0"}}
            #     ],
            #     "success": True
            # }
            xx = {"areaCount": {"a": 10, "b": 10, "c": 11, "d": 10},
                  "ageCount": {"46-60": "10%", "61": "15%", "31-45": "5%", "21-30": "50%", "0-20": "20%"},
                  "genderCount": {"men": {"num": 2, "percent": "10%"}, "women": {"num": 4, "percent": "20%"}},
                  "totalCount": f"{total_count}", "faceCount": 0,
                  "personCount": {"60-120": f"{q4_count}", "0-15": f"{q0_count}", "30-45": f"{q2_count}",
                                  "15-30": f"{q1_count}", ">120": f"{q5_count}",
                                  "45-60": f"{q3_count}"}}
            await websocket.send(json.dumps(xx))
        elif path.endswith('reportByHour'):
            xx = {"0": h0, "1": h1, "2": h2, "3": h3, "4": h4, "5": h5, "6": h6, "7": h7, "8": h8, "9": h9, "10": h10,
                  "11": h11, "12": h12, "13": h13, "14": h14, "15": h15, "16": h16, "17": h17, "18": h18, "19": h19,
                  "20": h20, "21": h21, "22": h22, "23": h23}
            await websocket.send(json.dumps(xx))
        elif path.endswith('realHeat'):
            # print(path, '333333333333')
            xx = []
            # realHeat = distinctlist(realHeat)
            for box in realHeat:
                # x1 = box[0] * w
                # x2 = box[2] * w
                # y1 = box[1] * h
                # y2 = box[3] * h
                x1 = box[0]
                y1 = box[1]
                bw = box[2]
                bh = box[3]
                xx.append({"x": int(x1 + bw / 2), "y": int(y1 + bh), "num": 1})
            await websocket.send(json.dumps(xx))

        else:
            # path.endswith('realTrack'):
            tstamp = time.time()
            while 1:
                for conn in connected:
                    # y = 315
                    # while 1:
                    realbox = realTrack['point']
                    realname = realTrack['id']
                    # with open('/dev/shm/realTrack', 'a') as fid:
                    #     fid.write(json.dumps(realTrack))
                    # print('*' * 99)
                    # print(realTrack)
                    # print('*' * 99)
                    '''
                    {'id': ['38048a3f', '8bab0388', '7e25fb59', '7f7bf5df', '842f9cfe', '68dd1072', '684baeef', '96026056', '5a35b0d3', 'fcf7b420'], 
                    'point': [[105.0, 278.0, 108.0, 197.0], [226.0, 209.0, 43.0, 237.0], [464.0, 213.0, 46.0, 70.0], [96.0, 186.0, 63.0, 137.0], [32.0, 186.0, 46.0, 84.0], 
                    [266.0, 100.0, 19.0, 65.0], [1.0, 149.0, 38.0, 84.0], [144.0, 197.0, 24.0, 52.0], [251.0, 108.0, 19.0, 52.0], [403.0, 120.0, 16.0, 65.0]]}
                    '''

                    jsondumpbatch = []
                    for i in range(len(realbox)):
                        bname = realname[i]
                        # bname = '1_136'
                        bname = f"1_{realname[i][0:3]}"
                        # for b in realbox[i]:
                        b = realbox[i]
                        # x1 = int(b[0] * yoloW)  # x
                        # x2 = int(b[2] * yoloW)  # w
                        # y1 = int(b[1] * yoloW)  # y
                        # y2 = int(b[3] * yoloW)  # h
                        x1, y1, bw, bh = b[0], b[1], b[2], b[3]
                        # jsondump = [
                        #     {"id": bname, "x": int((x1 + x2) / 2), "y": int((y1 + y2) / 2), "time": time.time()}]
                        jsondump = [
                            # {"id": bname, "x": int((x1 + bw / 2)), "y": int((y1 + bh / 2)), "time": 1608802281.56608}
                            {"id": bname, "x": int((x1 + bw / 2)), "y": int((y1 + bh / 2)), "time": time.time()}
                        ]

                        # await conn.send(json.dumps(jsondump))
                        # await asyncio.sleep(.8)
                        jsondumpbatch.extend(jsondump)

                    # enhance movment
                    # jsondump_enhance=[]
                    # for ppp_dict in jsondumpbatch:
                    #     print(ppp_dict)
                    #     xx1=ppp_dict['x']
                    #     yy1=ppp_dict['y']
                    #     nname=ppp_dict['id']
                    #     for ii in range(int(xx1), int((xx1) + np.random.randint(20)), 5):
                    #         jsondump_enhance.append({"id": nname, "x": int(ii + np.random.randint(-20, 20)),
                    #                              "y": int(int(yy1) + np.random.randint(-20, 20)),
                    #                              "time": time.time()})
                    # jsondumpbatch.extend(jsondump_enhance)
                    ##############

                    # await conn.send(json.dumps())
                    # await asyncio.sleep(.8)
                    # y += 5
                    # jsondumpbatch.extend(jsondump_enhance)

                    # print('Set changed size during iteration')
                    # await conn.recv()
                    # connected.clear()
                    # realTrack['point'].remove(realTrack['point'][i])
                    # realTrack['id'].remove(realTrack['id'][i])
                    try:
                        await conn.send(json.dumps(jsondumpbatch))
                        # await asyncio.sleep(.6)
                    except:
                        # print('Set changed size during iteration')
                        # print(realbox[i])
                        connected.remove(conn)
                        # raise
                        pass
                await asyncio.sleep(.6)

                if not len(connected):
                    break
                # await conn.recv()

                # await asyncio.sleep(1.)

                # for i in range(800, 235, -5):
                #     try:
                #         await conn.send(json.dumps([{"id": "1_136", "x": i, "y": y, "time": 1606874052384}]))
                #         await asyncio.sleep(1)
                #         y -= 5
                #     except RuntimeError:
                #         print('Set changed size during iteration')
                #         await conn.recv()
                #         connected.clear()
            # print(await conn.recv())
            # async for xx in websocket:
            #     print(xx)
            # await asyncio.sleep(1)
            # await asyncio.sleep(.1)
        # print(f"connections num :{len(connected)}")
    except:
        raise
        # pass
    finally:
        # connected.clear()
        pass

    # print(await websocket.recv())


def start_wsserver():
    asyncio.set_event_loop(asyncio.new_event_loop())
    # start_server = websockets.serve(server, HOST, 8888, ping_timeout=None)
    start_server = websockets.serve(server, HOST, 8888)

    asyncio.get_event_loop().run_until_complete(start_server)
    asyncio.get_event_loop().run_forever()


def get_found_gone(current_frame: np.ndarray, last_frame: np.ndarray, THRES=.7):
    assert current_frame.ndim == 2 & last_frame.ndim == 2
    current_frame, last_frame = current_frame.reshape(-1, 128), last_frame.reshape(-1, 128)
    cdistance = cosine_similarity(current_frame, last_frame)
    maxmat_0 = (np.max(cdistance, axis=0)[None, ...] == cdistance) & (cdistance > THRES)
    maxmat_1 = (np.max(cdistance, axis=1)[..., None] == cdistance) & (cdistance > THRES)
    indexes = np.where(maxmat_0 & maxmat_1)
    # indexes = np.where(cdistance > THRES)
    # found_current_index, found_last_index = np.unique(indexes[0]).reshape(-1), np.unique(indexes[1]).reshape(-1)
    found_current_index, found_last_index = indexes[0].reshape(-1), indexes[1].reshape(-1)
    assert found_current_index.size == found_last_index.size
    # np.take(a,found_c_i) np.delete(a,found_c_id
    # npfname=f'/dev/shm/{uuid.uuid4().hex}'
    # np.save(npfname+'_c', current_frame)
    # np.save(npfname+'_l',last_frame)
    # np.save(npfname+'_d',cdistance)
    # print(current_frame, '\n', last_frame, '\n', cdistance)
    return (np.sort(found_current_index), np.sort(found_last_index))
    # return (found_current_index, found_last_index)


def statis_to_ws(detsq: Queue):
    tWsServer = threading.Thread(target=start_wsserver, args=())
    tWsServer.start()
    cur_person_dict = {}
    global total_count, q0_count, q1_count, q2_count, q3_count, \
        q4_count, q5_count, h0, h1, h2, h3, h4, h5, \
        h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, \
        h16, h17, h18, h19, h20, h21, h22, h23, realTrack
    f, b, _ = detsq.get()
    # f, b, _ = f[0], b[0], d[0]
    f, b = f[0], b[0]
    last_person_dict = {
        'names': np.asarray([str(uuid.uuid4())[0:8] for _ in range(len(f))], dtype='<U8').reshape(1, -1),
        'features': np.asarray(f, dtype=np.float32).reshape(-1, 128),
        # 'boxes': np.asarray([x['box'] for x in b], dtype=np.float32).reshape(-1, 4),
        'boxes': np.asarray(b, dtype=np.float32).reshape(-1, 4),
        'timestamp': time.time()
    }
    # TODO maintain a total person features list ...
    all_people_features = np.zeros((1, 128), dtype=np.float32)
    all_people_features = np.vstack((all_people_features, last_person_dict['features']))
    all_people_features = np.unique(all_people_features, axis=0)
    # TODO END

    nowhour = datetime.datetime.now().hour
    if nowhour == 0:
        h0, h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, \
        h13, h14, h15, h16, h17, h18, h19, h20, h21, h22, h23 = \
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
    while 1:
        f, b, _ = detsq.get()
        f, b = f[0], b[0]  # imgs 2 img
        # print(f'f, b = f[0], b[0] {f}')
        # print(f'f, b = f[0], b[0] {b}')
        # f = [ndarray(128,)] b= [[471, 330, 126, 296],] d= [{'box': array([0.36846708, 0.45896153, 0.46698031, 0.87079132]),
        # 'class_name': 'person', 'score': 0.99644506, 'class_id': 0},]
        #######################################################################################################################
        current_frame_features = np.asarray(f, dtype=np.float32).reshape(-1, 128)
        # current_frame_boxes = np.asarray([x['box'] for x in b], dtype=np.float32).reshape(-1, 4)
        current_frame_boxes = np.asarray(b, dtype=np.float32).reshape(-1, 4)
        last_frame_features = last_person_dict['features']
        curr_found_idx, last_found_idx = get_found_gone(current_frame_features, last_frame_features, THRES=THRES)
        # print('*' * 88)
        # print(current_frame_features.shape, last_frame_features.shape)  # (10, 128) (9, 128)
        # print(curr_found_idx, last_found_idx)  # [0 1 2 3 4 5 6 7 8 9] [0 1 2 3 4 5 6 7 8]
        # print(last_person_dict['names'],
        #       last_found_idx)  # ['9c41ea8e' 'f5b56538' 'c08b1ca0' '923220ae' 'fbc33b7e' '9b020cd0' '168bfad2' 'abdaf773' '1aa0e116'] [0 1 2 3 4 5 6 7 8]
        namefound = np.take(last_person_dict['names'], last_found_idx, axis=1)

        current_frame_boxes_found = np.take(current_frame_boxes, curr_found_idx, axis=0)
        current_frame_boxes_new = np.delete(current_frame_boxes, curr_found_idx, axis=0)
        featuresnew = np.delete(current_frame_features, curr_found_idx, axis=0)
        namesnew = np.asarray([str(uuid.uuid4())[0:8] for _ in range(len(featuresnew))], dtype='<U8').reshape(1, -1)
        # print(last_person_dict['names'], last_found_idx, namesnew)
        # os._exit(0)
        #######################################################################################################################

        realTrack['id'] = namefound.reshape(-1).tolist()
        realTrack['id'].extend(namesnew.reshape(-1).tolist())
        realTrack['point'] = current_frame_boxes_found.tolist()
        realTrack['point'].extend(current_frame_boxes_new.tolist())
        # print(realTrack)
        # realTrack['point'] = boxesfound.tolist()
        # realTrack['point'].extend(boxesnew.tolist())
        # print(realTrack)
        # maskgone = np.ones(len(last_person_dict['features']), dtype=bool)
        # maskgone[found] = False
        # resultgone = last_person_dict['features'][maskgone, ...]

        #########################################################
        resultgone = np.delete(last_frame_features, last_found_idx, axis=0)
        # print('-' * 99)
        # print(f'curr fea length is {len(current_frame_features)}, last fea length is {len(last_frame_features)}')
        # print(f'curr_found_idx is {curr_found_idx}, last_found_idx is {last_found_idx}')
        # print(f'namefound is {namefound}, realTrack point is {current_frame_boxes}')
        # print(f'resultgone shape is {resultgone.shape}, len is {len(resultgone)}')
        # if len(resultgone) > 2:
        #     npfname=f'/dev/shm/{uuid.uuid4().hex}'
        #     np.save(npfname+'_c', current_frame_features)
        #     np.save(npfname+'_l',last_frame_features)
        #     os._exit(0)
        #########################################################

        total_count += min(1, len(resultgone))
        last_person_dict = {
            'names': np.hstack((namefound, namesnew)).reshape(1, -1),
            # 'features': np.vstack((featuresfound, featuresnew)).reshape(-1, 128),
            'features': current_frame_features,
            # 'boxes': np.vstack((boxesfound, boxesnew)).reshape(-1, 4),
            'boxes': current_frame_boxes,
            'timestamp': time.time()
        }
        # print('*' * 88)
        # print(total_count, len(resultgone))
        # print('*' * 88)
        # print(last_person_dict)
        # print('*' * 88)

        # realheatnew = np.asarray(last_person_dict['boxes'], dtype=np.int8).reshape(-1, 4)
        # realHeatold = np.asarray(realHeat, dtype=np.int8).reshape(-1, 4)
        # realheattmp = np.vstack((realheatnew, realHeatold))
        # # realHeat.clear()
        # for box in np.unique(realheattmp, axis=0):
        #     realHeat.append(box)
        # for box in last_person_dict['boxes']:
        realHeat.clear()
        for box in current_frame_boxes:
            realHeat.append(box)
        if nowhour == 0:
            h0 += len(resultgone)
        elif nowhour == 1:
            h1 += len(resultgone)
        elif nowhour == 2:
            h2 += len(resultgone)
        elif nowhour == 3:
            h3 += len(resultgone)
        elif nowhour == 4:
            h4 += len(resultgone)
        elif nowhour == 5:
            h5 += len(resultgone)
        elif nowhour == 6:
            h6 += len(resultgone)
        elif nowhour == 7:
            h7 += len(resultgone)
        elif nowhour == 8:
            h8 += len(resultgone)
        elif nowhour == 9:
            h9 += len(resultgone)
        elif nowhour == 10:
            h10 += len(resultgone)
        elif nowhour == 11:
            h11 += len(resultgone)
        elif nowhour == 12:
            h12 += len(resultgone)
        elif nowhour == 13:
            h13 += len(resultgone)
        elif nowhour == 14:
            h14 += len(resultgone)
        elif nowhour == 15:
            h15 += len(resultgone)
        elif nowhour == 16:
            h16 += len(resultgone)
        elif nowhour == 17:
            h17 += len(resultgone)
        elif nowhour == 18:
            h18 += len(resultgone)
        elif nowhour == 19:
            h19 += len(resultgone)
        elif nowhour == 20:
            h20 += len(resultgone)
        elif nowhour == 21:
            h21 += len(resultgone)
        elif nowhour == 22:
            h22 += len(resultgone)
        elif nowhour == 23:
            h23 += len(resultgone)

        # try:
        #     while 1:
        #         detsq.get_nowait()
        # except:
        #     pass
        # time.sleep(.3)


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

    ffmpegprocess = (
        ffmpeg
            .input('pipe:', re=None)
            # .filter('scale', width='570', height='320')
            .output('rtmp://192.168.8.121/live/bbb', vcodec='copy', format='flv', r='25'
                    # ,s='570x320'
                    )
            .run_async(pipe_stdin=True))

    input_frame_q = Queue(maxsize=200)
    # input_frame_q = Queue()
    # outq = Queue(maxsize=2000)
    detstatisq = Queue()
    boxq = Queue()
    #############
    # use c so for read frame
    # gcf = Process(target=get_camera_frame, args=(None,))
    # gcf.start()
    #############

    pm_detect_0 = Process(target=pintmain_detection, args=(input_frame_q, boxq))
    pm_detect_0.start()
    # pm_detect_1 = Process(target=pintmain_detection, args=(input_frame_q, boxq))
    # pm_detect_1.start()
    pm = Process(target=pintmain, args=(input_frame_q, boxq, detstatisq))
    pm.start()
    # print('aaaaaaaaaaaaaaa')
    # gci = Process(target=gen_camera_frame,args=(None,))
    # gcq = Process(target=p_getframe_python_to_queue, args=(q,))

    # for cut det and fea
    # gpq = Process(target=p_putframe_python_to_pipe, args=(input_frame_q, outq))
    gpq = Process(target=p_putframe_python_to_pipe, args=(input_frame_q, boxq))

    # print('bbbbbbbbbbbbbbb')
    # gcq.start()
    gpq.start()
    # print('cccccccccccc')
    statisprocess = Process(target=statis_to_ws, args=(detstatisq,))
    statisprocess.start()
    # print('ddddddddddd')
    # fire.Fire(demo_classer)
    # gcf.join()
    # pm.join()
    # for read video from local ./videos/*.avi
    for video in cycle(glob.glob("./videos/*.avi")):
        cap = cv2.VideoCapture(video)
        while cap.isOpened():
            if cap.grab():
                flag, frame = cap.retrieve()
                if not flag:
                    continue
                else:
                    input_frame_q.put(frame)
            else:
                print('grab error ......')
                break
        cap.release()
'''
 ffmpeg -hwaccel_device /dev/dri/card0 -r 9 -i /dev/shm/rtmp_q -an -s 320*240  -f flv -r 9 rtmp://192.168.8.121/live/bbb -c:v h264_vaapi
  ffmpeg -hwaccel_device /dev/dri/card0 -r 9 -i /dev/shm/rtmp_q -an -s 320*240  -f flv -r 9 rtmp://192cvtColor.168.8.121/live/bbb -c:v hevc_vaapi
  -use_wallclock_as_timestamps 1
  -re
'''
