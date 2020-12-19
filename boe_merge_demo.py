import cv2
import os
import json
import time
import uuid
import numpy as np
from itertools import cycle

import sqlite3
from multiprocessing import Process
from api.CameraApi import (
    get_ins,
    start
)
import multiprocessing
import threading
from multiprocessing import Process, Queue
import ffmpeg
from api.convertimg import gen_camera_frame
import fire
from api.yolo_reid_pint_api import YoloReidPintAPIAlgorithm as pint_api
from api.yolo_reid_pint_api import Config
from api.yolo_reid_pint_api import cosine_distance
from api.plot_utils import plot_multi_images
from api.timer import Timer
import sqlite3
import asyncio
import websockets
import datetime
import glob

IMAGE_PIPE = '/dev/shm/img_q'
RTMP_PIPE = '/dev/shm/rtmp_q'
w = 640
h = 480
c = 3
THRES = .15
THRES_ALL = .7
HOST = "0.0.0.0"
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


def p_putframe_python_to_pipe(q: Queue, outq: Queue):
    while not os.path.exists(RTMP_PIPE):
        print(f'wait for frame pipe : {RTMP_PIPE}')
        time.sleep(1)

    def put_real_stream(outfifo, q):
        while 1:
            try:
                i = q.get()
                ret2, frame2 = cv2.imencode('.png', i)
                outfifo.write(frame2)
                # outfifo.write(i)
            except:
                time.sleep(.5)
                print(f'none q img')
                # raise

    def put_det_stream(outfifo, outq, realq):
        while 1:
            # outi = outq.get()
            detect_result = outq.get()
            img = realq.get()
            try:
                if img.shape[2]==3:
                    plot_multi_images([img], detect_result)

                    ret2, frame3 = cv2.imencode('.png', img)
                    outfifo.write(frame3)
            except:
                pass
            # outfifo.write(img)

    with open(RTMP_PIPE, 'wb') as outfifo:
        t1 = threading.Thread(target=put_real_stream, args=(outfifo, q))
        t2 = threading.Thread(target=put_det_stream, args=(outfifo, outq, q))
        # t1 = multiprocessing.Process(target=put_real_stream, args=(outfifo, q))
        # t2 = multiprocessing.Process(target=put_det_stream, args=(outfifo, outq, q))
        t1.start()
        t2.start()
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
        print(f'pth is :{path}')
        if path.endswith('reportDetail'):
            print(path, '111111111111')
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
                  # "personCount": {"60-120": "50", "0-15": "10", "30-45": "30", "15-30": "20", ">120": "60",
                  #                 "45-60": "40"}},
                  "personCount": {"60-120": f"{q4_count}", "0-15": f"{q0_count}", "30-45": f"{q2_count}",
                                  "15-30": f"{q1_count}", ">120": f"{q5_count}",
                                  "45-60": f"{q3_count}"}}
            await websocket.send(json.dumps(xx))
        elif path.endswith('reportByHour'):
            print(path, '22222222222222')
            # xx = {
            #     "code": 200,
            #     "data": {
            #         "bankName": "ICBC",
            #         "times": {
            #             "0-15": 6626,
            #             "15-30": 0,
            #             "30-45": 0,
            #             "45-60": 0,
            #             "60-120": 0,
            #             ">120": 0
            #         },
            #         "aresUser": {
            #             "A": 15485,
            #             "B": 22217,
            #             "C": 31327,
            #             "D": 13794
            #         },
            #         "hoursUser": {
            #             "<8:00": 0,
            #             "8:00-10:00": 968,
            #             "10:00-12:00": 1894,
            #             "12:00-14:00": 1900,
            #             "14:00-16:00": 1893,
            #             "16:00-18:00": 830,
            #             ">18:00": 0
            #         },
            #         "totalUser": 7485,
            #         "aveUser": 7485
            #     },
            #     "sucdess": True
            # }
            # xx = {"0": 0, "1": 0, "2": 0, "3": 0, "4": 0, "5": 0, "6": 0, "7": 0, "8": 7, "9": 14, "10": 10,
            #       "11": 11, "12": 12, "13": 13, "14": 14, "15": 15, "16": 16, "17": 17, "18": 18, "19": 19,
            #       "20": 20, "21": 21, "22": 22}
            xx = {"0": h0, "1": h1, "2": h2, "3": h3, "4": h4, "5": h5, "6": h6, "7": h7, "8": h8, "9": h9, "10": h10,
                  "11": h11, "12": h12, "13": h13, "14": h14, "15": h15, "16": h16, "17": h17, "18": h18, "19": h19,
                  "20": h20, "21": h21, "22": h22, "23": h23}
            await websocket.send(json.dumps(xx))
        elif path.endswith('realHeat'):
            print(path, '333333333333')
            xx = []
            # realHeat = distinctlist(realHeat)
            for box in realHeat:
                x1 = box[0] * w
                x2 = box[2] * w
                y1 = box[1] * h
                y2 = box[3] * h
                xx.append({"x": (x1 + x2) / 2, "y": (y1 + y2) / 2, "num": 1})

            # xx = [{"x": 235, "y": 315, "num": 4}, {"x": 100, "y": 100, "num": 4},
            #       {"x": 300, "y": 300, "num": 4}, {"x": 1000, "y": 1000, "num": 4}, ]
            # print(json.dumps(xx))
            await websocket.send(json.dumps(xx))

        else:
            # path.endswith('realTrack'):
            tstamp = time.time()
            print(path, '4444444444')
            xx = {
                "code": 200,
                "data": [
                    {"id": "1", "x": 2750, "y": 9750, "time": 1541430823000},
                    {"id": "2", "x": 1250, "y": 4750, "time": 1541430823000},
                    {"id": "3", "x": 2750, "y": 6250, "time": 1541430823000},
                    {"id": "4", "x": 3250, "y": 5250, "time": 1541430823000},
                    {"id": "5", "x": 2250, "y": 10750, "time": 1541430823000},
                    {"id": "5", "x": 2250, "y": 10750, "time": 1541430823002},
                    {"id": "1", "x": 2750, "y": 10250, "time": 1541430823002},
                    {"id": "2", "x": 1250, "y": 5250, "time": 1541430823002},
                    {"id": "3", "x": 2750, "y": 6250, "time": 1541430823002},
                    {"id": "4", "x": 3250, "y": 5750, "time": 1541430823002},
                    {"id": "1", "x": 2750, "y": 9750, "time": 1541430823004},
                    {"id": "2", "x": 1250, "y": 4750, "time": 1541430823004},
                    {"id": "3", "x": 2750, "y": 6250, "time": 1541430823004}
                ],
                "sucdess": True
            }
            xx = [{"id": "1_136", "x": 235, "y": 315, "time": 1606874052384},
                  {"id": "1_136", "x": 280, "y": 390, "time": 1606874052384}]
            # while 1:
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
                for i in range(len(realbox)):
                    try:
                        bname = realname[i]
                        # bname = '1_136'
                        bname = f"1_{realname[i][0][:4]}"
                        # for b in realbox[i]:
                        b = realbox[i]
                        x1 = int(b[0] * w)
                        x2 = int(b[2] * w)
                        y1 = int(b[1] * h)
                        y2 = int(b[3] * h)
                        jsondump = [
                            {"id": bname, "x": int((x1 + x2) / 2), "y": int((y1 + y2) / 2), "time": 1606874052384}]
                        await conn.send(json.dumps(jsondump))
                        await asyncio.sleep(.8)

                        # enhance movment
                        for i in range(int((x1 + x2) / 2), int((x1 + x2) / 2) + np.random.randint(50), 5):
                            try:
                                await conn.send(
                                    json.dumps([{"id": bname, "x": i,
                                                 "y": int((y1 + y2) / 2 + np.random.randint(-10, 10)),
                                                 "time": 1606874052384}]))
                                await asyncio.sleep(.8)
                                # y += 5
                            except RuntimeError:
                                pass

                                # print('Set changed size during iteration')
                                # await conn.recv()
                                # connected.clear()
                        # realTrack['point'].remove(realTrack['point'][i])
                        # realTrack['id'].remove(realTrack['id'][i])
                    except:
                        # print('Set changed size during iteration')
                        # print(realbox[i])
                        connected.remove(conn)
                        # raise
                        continue
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
        pass
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


def cosine_distance_fine(a, b):
    return a @ b.T


def statis_to_ws(detsq: Queue):
    tWsServer = threading.Thread(target=start_wsserver, args=())
    tWsServer.start()
    cur_person_dict = {}
    global total_count, q0_count, q1_count, q2_count, q3_count, \
        q4_count, q5_count, h0, h1, h2, h3, h4, h5, \
        h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, \
        h16, h17, h18, h19, h20, h21, h22, h23, realTrack
    f, b, d = detsq.get()
    f, b, d = f[0], b[0], d[0]
    # print(f, b, d)
    # last_person_dict=dict(zip([str(uuid.uuid4())[0:8] for _ in range(len(f))], f))
    last_person_dict = {
        'names': np.asarray([str(uuid.uuid4())[0:8] for _ in range(len(f))], dtype='<U8').reshape(-1, 1),
        'features': np.asarray(f, dtype=np.float32).reshape(-1, 128),
        'boxes': np.asarray([x['box'] for x in d], dtype=np.float32).reshape(-1, 4),
        'timestamp': time.time()
    }
    all_people_features = np.zeros((1, 128), dtype=np.float32)
    all_people_features = np.vstack((all_people_features, last_person_dict['features']))
    all_people_features = np.unique(all_people_features, axis=0)
    nowhour = datetime.datetime.now().hour
    nowminites = datetime.datetime.now().minute
    if nowhour == 0:
        h0, h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, \
        h13, h14, h15, h16, h17, h18, h19, h20, h21, h22, h23 = \
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
    if nowminites == 0:
        realHeat.clear()
    while 1:
        f, b, d = detsq.get()
        # print('*' * 99)
        # print(type(f), len(f))
        # print(f, b, d)
        # os._exit(0)
        f, _, d = f[0], b[0], d[0]  # imgs 2 img

        # f = [ndarray(128,)] b= [[471, 330, 126, 296],] d= [{'box': array([0.36846708, 0.45896153, 0.46698031, 0.87079132]),
        # 'class_name': 'person', 'score': 0.99644506, 'class_id': 0},]
        found = []
        newones = []
        for i in range(len(f)):
            # print('*'*99)
            # print(type(f),len(f))
            # print(last_person_dict)
            # os._exit(0)

            # if current people is found in history all features
            # all peple found start
            currentfeature = np.asarray(f[i], dtype=np.float32).reshape(-1, 128)
            # distince_from_allpeople = cosine_distance(currentfeature, all_people_features).reshape(-1)
            # argmax_all = np.argmax(distince_from_allpeople)
            # # print('-' * 88)
            # # print('people found from history all list....', all_people_features.shape, distince_from_allpeople)
            # # print('-' * 88)
            # if distince_from_allpeople[argmax_all] > THRES_ALL:
            #     # print('+' * 88)
            #     # print('people found from history all list....', all_people_features.shape, distince_from_allpeople)
            #     # print('+' * 88)
            #     continue
            # all_people_features = np.vstack((all_people_features, currentfeature))
            # all_people_features = np.unique(all_people_features, axis=0)
            # all peple found end
            ##################

            cdistance = cosine_distance_fine(currentfeature,
                                             last_person_dict['features']).reshape(-1)
            # cdistance (len(last_person_number),)
            # argmin = np.argmin(np.abs(cdistance))
            # argmin = np.argmin(cdistance)
            argmax = np.argmax(cdistance)
            # print('*' * 88)
            # print(cdistance, argmax)
            # print('*' * 88)
            try:
                if cdistance[argmax] > THRES:
                    found.append(argmax)
                    # print(found, newones)
                    continue
            except:
                print('*' * 88)
                print(cdistance, argmax, last_person_dict['features'], currentfeature)
                print('*' * 88)
                raise
            newones.append(i)

        maskfound = np.zeros(len(last_person_dict['features']), dtype=bool)
        maskfound[found] = True
        featuresfound = last_person_dict['features'][maskfound, ...]
        boxesfound = last_person_dict['boxes'][maskfound, ...]
        namefound = last_person_dict['names'][maskfound, ...]

        masknew = np.zeros(len(f), dtype=bool)
        masknew[newones] = True
        featuresnew = np.asarray(f, dtype=np.float32).reshape(-1, 128)[masknew, ...]
        namesnew = np.asarray([str(uuid.uuid4())[0:8] for _ in range(len(featuresnew))], dtype='<U8').reshape(-1, 1)
        boxesnew = np.asarray([x['box'] for x in d], dtype=np.float32).reshape(-1, 4)[masknew, ...]

        # realTrack['id'].extend(namefound.tolist())
        # realTrack['id'].extend(namesnew.tolist())
        # realTrack['point'].extend(boxesfound.tolist())
        # realTrack['point'].extend(boxesnew.tolist())

        realTrack['id'] = namefound.tolist()
        realTrack['id'].extend(namesnew.tolist())
        realTrack['point'] = boxesfound.tolist()
        realTrack['point'].extend(boxesnew.tolist())
        # print(realTrack)
        maskgone = np.ones(len(last_person_dict['features']), dtype=bool)
        maskgone[found] = False
        resultgone = last_person_dict['features'][maskgone, ...]
        # namegone = last_person_dict['names'][maskgone, ...]

        total_count += len(resultgone)

        name = str(uuid.uuid4())[0:8]
        last_person_dict = {
            'names': np.vstack((namefound, namesnew)).reshape(-1, 1),
            'features': np.vstack((featuresfound, featuresnew)).reshape(-1, 128),
            'boxes': np.vstack((boxesfound, boxesnew)).reshape(-1, 4),
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
        for box in last_person_dict['boxes']:
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
        # time.sleep(1)


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

    input_frame_q = Queue(maxsize=2000)
    outq = Queue(maxsize=2000)
    detstatisq = Queue()

    #############
    # use c so for read frame
    # gcf = Process(target=get_camera_frame, args=(None,))
    # gcf.start()
    #############

    pm = Process(target=pintmain, args=(input_frame_q, outq, detstatisq))
    pm.start()

    # gci = Process(target=gen_camera_frame,args=(None,))
    # gcq = Process(target=p_getframe_python_to_queue, args=(q,))

    gpq = Process(target=p_putframe_python_to_pipe, args=(input_frame_q, outq))

    # gcq.start()
    gpq.start()

    statisprocess = Process(target=statis_to_ws, args=(detstatisq,))
    statisprocess.start()
    # fire.Fire(demo_classer)
    # gcf.join()
    # pm.join()

    # for read video from local ./videos/*.avi
    for video in cycle(glob.glob("./videos/*.avi")):
        cap = cv2.VideoCapture(video)
        f = True
        try:
            while cap.isOpened() & f:
                f, i = cap.read()
                # resized = cv2.resize(i, (w, h), interpolation=cv2.INTER_CUBIC)
                try:
                    # if i.shape[2] == 3:
                    input_frame_q.put(i)
                except:
                    raise
        except:
            raise

'''
 ffmpeg -hwaccel_device /dev/dri/card0 -r 9 -i /dev/shm/rtmp_q -an -s 320*240  -f flv -r 9 rtmp://192.168.8.121/live/bbb -c:v h264_vaapi
  ffmpeg -hwaccel_device /dev/dri/card0 -r 9 -i /dev/shm/rtmp_q -an -s 320*240  -f flv -r 9 rtmp://192.168.8.121/live/bbb -c:v hevc_vaapi
'''
