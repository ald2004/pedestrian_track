import asyncio
import websockets
import datetime
import random
import json

HOST = "192.168.8.121"
connected = set()
'''
    ws://192.168.8.121/socket/reportDetail    ok
    {
    “totalCount”:100,    
    “areaCount”:,{“areaA”:33,”areaB”:333}
    “personCount”: {“12”:33,”13”:333} 
    }
'''


async def server(websocket, path: str):
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
                  "totalCount": 31, "faceCount": 0,
                  "personCount": {"60-120": "0", "0-15": "18", "30-45": "0", "15-30": "0", ">120": "0",
                                  "45-60": "0"}}
            await websocket.send(json.dumps(xx))
        elif path.endswith('reportByHour'):
            print(path, '22222222222222')
            xx = {
                "code": 200,
                "data": {
                    "bankName": "ICBC",
                    "times": {
                        "0-15": 6626,
                        "15-30": 0,
                        "30-45": 0,
                        "45-60": 0,
                        "60-120": 0,
                        ">120": 0
                    },
                    "aresUser": {
                        "A": 15485,
                        "B": 22217,
                        "C": 31327,
                        "D": 13794
                    },
                    "hoursUser": {
                        "<8:00": 0,
                        "8:00-10:00": 968,
                        "10:00-12:00": 1894,
                        "12:00-14:00": 1900,
                        "14:00-16:00": 1893,
                        "16:00-18:00": 830,
                        ">18:00": 0
                    },
                    "totalUser": 7485,
                    "aveUser": 7485
                },
                "sucdess": True
            }
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


start_server = websockets.serve(server, HOST, 8888, ping_timeout=None)

asyncio.get_event_loop().run_until_complete(start_server)
asyncio.get_event_loop().run_forever()

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


