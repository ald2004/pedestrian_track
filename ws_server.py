import os
import asyncio
import websockets
import datetime
import random
import json
import sqlite3
import threading
from contextlib import closing
import numpy as np
from datetime import datetime
import time

from api.yolo_reid_pint_api import cosine_distance
from sklearn.metrics.pairwise import cosine_similarity

HOST = "0.0.0.0"
connected = set()
'''
    ws://192.168.8.121/socket/reportDetail    ok
    {
    “totalCount”:100,    
    “areaCount”:,{“areaA”:33,”areaB”:333}
    “personCount”: {“12”:33,”13”:333} 
    }
'''

total_count, area_a, area_b, area_c, area_d, \
q0_count, q1_count, q2_count, q3_count, \
q4_count, q5_count = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
PERSON_THRES = 1.0
a0_count, a1_count, a2_count, a3_count, a4_count, \
a5_count, a6_count, a7_count, a8_count, a9_count, aa_count, ab_count, \
ac_count, ad_count, ae_count, af_count, b0_count, b1_count, b2_count, \
b3_count, b4_count, b5_count, b6_count, b7_count = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0


def cosine_similarity(X, Y=None, dense_output=True):
    X_normalized = np.asarray(X) / np.linalg.norm(X, axis=0, keepdims=True)
    Y_normalized = np.asarray(Y) / np.linalg.norm(Y, axis=0, keepdims=True)
    if X_normalized.ndim > 2 or Y_normalized.ndim > 2:
        ret = np.dot(X_normalized, Y_normalized.T)
    else:
        ret = X_normalized @ Y_normalized.T
    return np.abs(ret)


async def server(websocket, path: str):
    global total_count, area_a, area_b, area_d, area_d, \
        q0_count, q1_count, q2_count, q3_count, \
        q4_count, q5_count, a0_count, a1_count, a2_count, a3_count, a4_count, \
        a5_count, a6_count, a7_count, a8_count, a9_count, aa_count, ab_count, \
        ac_count, ad_count, ae_count, af_count, b0_count, b1_count, b2_count, \
        b3_count, b4_count, b5_count, b6_count, b7_count
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
            xx = {"areaCount": {"A": 3938, "B": 0, "C": 11, "D": 0},
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
            xx = {"0": 0, "1": 0, "2": 0, "3": 0, "4": 0, "5": 0, "6": 0, "7": 0, "8": 7, "9": 14, "10": 10,
                  "11": 11, "12": 12, "13": 13, "14": 14, "15": 15, "16": 16, "17": 17, "18": 18, "19": 19,
                  "20": 20, "21": 21, "22": 22}
            await websocket.send(json.dumps(xx))
        elif path.endswith('realHeat'):
            print(path, '333333333333')
            xx = [{"x": 235, "y": 315, "num": 4}, {"x": 100, "y": 100, "num": 4},
                  {"x": 300, "y": 300, "num": 4}, {"x": 1000, "y": 1000, "num": 4}, ]
            await websocket.send(json.dumps(xx))
        else:
            # path.endswith('realTrack'):
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
                y = 315
                while 1:
                    for i in range(235, 800, 5):
                        try:
                            await conn.send(json.dumps([{"id": "1_136", "x": i, "y": y, "time": 1606874052384}]))
                            await asyncio.sleep(1)
                            y += 5
                        except RuntimeError:
                            print('Set changed size during iteration')
                            connected.clear()
                    for i in range(800, 235, -5):
                        try:
                            await conn.send(json.dumps([{"id": "1_136", "x": i, "y": y, "time": 1606874052384}]))
                            await asyncio.sleep(1)
                            y -= 5
                        except RuntimeError:
                            print('Set changed size during iteration')
                            connected.clear()
            # print(await conn.recv())
            # async for xx in websocket:
            #     print(xx)
            # await asyncio.sleep(1)
            # await asyncio.sleep(.1)
        # print(f"connections num :{len(connected)}")
    finally:
        # connected.clear()
        pass

    # print(await websocket.recv())


def start_wsserver():
    asyncio.set_event_loop(asyncio.new_event_loop())
    start_server = websockets.serve(server, HOST, 8888, ping_timeout=None)
    asyncio.get_event_loop().run_until_complete(start_server)
    asyncio.get_event_loop().run_forever()


def statis_data():
    global total_count, area_a, area_b, area_d, area_d, \
        q1_count, q2_count, q3_count, \
        q4_count, q5_count, a0_count, a1_count, a2_count, a3_count, a4_count, \
        a5_count, a6_count, a7_count, a8_count, a9_count, aa_count, ab_count, \
        ac_count, ad_count, ae_count, af_count, b0_count, b1_count, b2_count, \
        b3_count, b4_count, b5_count, b6_count, b7_count
    pre_frame_pedests = {
        'f': np.empty((1, 128), dtype=np.float32),
        't': np.empty((1), dtype=np.float)
    }
    while 1:
        with closing(sqlite3.connect("aquarium.sqlite")) as connection:
            with closing(connection.cursor()) as cursor:
                with open("pedes_statis.sql", "rb") as sql_file:
                    sql_as_string = sql_file.read()
                    cursor.executescript(sql_as_string.decode())
                # try:
                #     with open("pedes_statis.sql") as sql_file:
                #         sql_as_string = sql_file.read()
                #         cursor.executescript(sql_as_string.decode())
                # except:
                #     pass

                # get statis data
                maxid, minid, total_count, \
                area_a, area_b, area_c, area_d, \
                q1_count, q2_count, q3_count, \
                q4_count, q5_count, a0_count, a1_count, a2_count, a3_count, a4_count, \
                a5_count, a6_count, a7_count, a8_count, a9_count, aa_count, ab_count, \
                ac_count, ad_count, ae_count, af_count, b0_count, b1_count, b2_count, \
                b3_count, b4_count, b5_count, b6_count, b7_count, b8_count, = \
                    cursor.execute("select max(id),min(id),max(total_count),"
                                   "max(area_a),max(area_b),max(area_c),max(area_d),"
                                   "max(q1_count),max(q2_count),max(q3_count),max(q4_count),"
                                   "max(q5_count),max(a0_count),max(a1_count),max(a2_count),"
                                   "max(a3_count),max(a4_count),max(a5_count),max(a6_count),"
                                   "max(a7_count),max(a8_count),max(a9_count),max(aa_count),"
                                   "max(ab_count),max(ac_count),max(ad_count),max(ae_count),"
                                   "max(af_count),max(b0_count),max(b1_count),max(b2_count),"
                                   "max(b3_count),max(b4_count),max(b5_count),max(b6_count),max(b7_count),max(b8_count) "
                                   "from pedes_statis ").fetchall()[0]

                # get data from pedetrain
                pMaxid, pMinid = cursor.execute("select max(id),min(id) from pedestrian").fetchall()[0]
                for rowid in range(pMinid, pMaxid, 200):  # every frame
                    current_frame_pedests = {
                        'f': np.empty((1, 128), dtype=np.float32),
                        't': np.empty((1), dtype=np.float)
                    }
                    for _, b, f, d, t in cursor.execute(
                            "select id,boxes,features,detect_result,timestamp from pedestrian where id < ?",
                            (rowid,)).fetchall():
                        # tbox = np.frombuffer(b, dtype=np.int)
                        tfeatures = np.frombuffer(f, dtype=np.float32)
                        # tresult = json.loads(d)
                        ttime = datetime.strptime(t, '%Y%m%d %H:%M:%S').timestamp()
                        try:
                            current_frame_pedests['f'] = np.vstack((current_frame_pedests['f'], tfeatures))
                            current_frame_pedests['t'] = np.vstack((current_frame_pedests['t'], ttime))
                        except:
                            continue
                    # compute similarity
                    pre_f = pre_frame_pedests['f'][1:]
                    cur_f = current_frame_pedests['f'][1:]
                    if len(pre_f):
                        print(pre_f)
                        print('==' * 80)
                        print(cur_f)
                        os._exit(0)
                    if len(pre_f) and len(cur_f):
                        print('*'*300)
                        # distmatrix = cosine_similarity(pre_f, cur_f)
                        distmatrix = cosine_distance(pre_f,cur_f)
                        # np.unravel_index(np.argmin(distmatrix, axis=None), distmatrix.shape)
                        p_c_ind = np.argwhere(distmatrix > PERSON_THRES)
                        total_count += np.abs(len(pre_f) - len(p_c_ind))
                        print(distmatrix)
                        print(total_count)
                        print(p_c_ind)
                        for awayperidx, awayfeature in enumerate(pre_f):
                            if awayperidx in p_c_ind[:, 0]:
                                current_frame_pedests['t'][awayperidx + 1] = pre_frame_pedests['t'][
                                    p_c_ind[awayperidx, 0] + 1]
                            else:
                                qq = time.time() - pre_frame_pedests['t'][awayperidx + 1]
                                if qq >= 7200:  # q5
                                    q5_count += 1
                                elif qq >= 3600:  # q4
                                    q4_count += 1
                                elif qq >= 2700:  # 3
                                    q3_count += 1
                                elif qq >= 2700:  # q2
                                    q2_count += 1
                                else:  # q1s
                                    q1_count += 1
                                now = datetime.now()
                                if now == 0:
                                    a0_count += 1
                                elif now == 1:
                                    a1_count += 1
                                elif now == 2:
                                    a2_count += 1
                                elif now == 3:
                                    a3_count += 1
                                elif now == 4:
                                    a4_count += 1
                                elif now == 5:
                                    a5_count += 1
                                elif now == 6:
                                    a6_count += 1
                                elif now == 7:
                                    a7_count += 1
                                elif now == 8:
                                    a8_count += 1
                                elif now == 9:
                                    a9_count += 1
                                elif now == 10:
                                    aa_count += 1
                                elif now == 11:
                                    a1_count += 1
                                elif now == 12:
                                    ab_count += 1
                                elif now == 13:
                                    ac_count += 1
                                elif now == 14:
                                    ad_count += 1
                                elif now == 15:
                                    ae_count += 1
                                elif now == 16:
                                    af_count += 1
                                elif now == 17:
                                    b1_count += 1
                                elif now == 18:
                                    b2_count += 1
                                elif now == 19:
                                    b3_count += 1
                                elif now == 20:
                                    b4_count += 1
                                elif now == 21:
                                    b5_count += 1
                                elif now == 22:
                                    b6_count += 1
                                else:
                                    b7_count += 1

                        # for ci in p_c_ind[:, 1]:
                        #     pass
                    pre_frame_pedests['f'] = current_frame_pedests['f']
                    pre_frame_pedests['t'] = current_frame_pedests['t']

        time.sleep(2)

        records = [(1, 'Glen', 8),
                   (2, 'Elliot', 9),
                   (3, 'Bob', 7)]
        # insert multiple records in a single query
        c.executemany('INSERT INTO students VALUES(?,?,?);', records);
        replacesql = "INSERT OR REPLACE INTO table(column_list) VALUES(value_list);"


if __name__ == '__main__':
    tWsServer = threading.Thread(target=start_wsserver, args=())
    tWsServer.start()

    tStatic = threading.Thread(target=statis_data, args=())
    tStatic.start()

    # pWsServer.join()
# async def time(websocket, path):

# while True:
#     print(f"< path is : {path}")
#     now = datetime.datetime.utcnow().isoformat() + "Z"
#     await websocket.send(now)
#     await asyncio.sleep(random.random() * 3)


# start_server = websockets.serve(time, "127.0.0.1", 5678)
#
# asyncio.get_event_loop().run_until_complete(start_server)
# asyncio.get_event_loop().run_forever()
