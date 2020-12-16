import sqlite3
import numpy as np
import os
import json
import threading
import time
import glob
from contextlib import closing


def read_npz_to_db(conn, cur):
    os.chdir('/dev/shm')
    filenames = glob.glob('/dev/shm/*.npz')
    # cur.execute("delete from pedestrian")
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
    for filename in filenames:
        # '/dev/shm/1607657453.1121302.npz'
        # print(filenames) yyyyMMdd HH:mm:ss
        y, m, d, hh, mm, ss, weekday, jday, dst = time.localtime(float(filename.split('/')[-1][:-4]))
        timestring = f"{y}{m}{d} {hh}:{mm}:{ss}"
        try:
            compressed = np.load(filename, allow_pickle=True)
            features, boxes, detect_result = compressed['f'][0], compressed['b'][0], compressed['d'][0]
        except:
            print("zipfile.BadZipFile: File is not a zip file")
            # os.rename(filename, filename + '.badzip')
            os.remove(filename)
            continue

        # print(len(features), le*' * 88)
        #         # print(timestrin(boxes), len(detect_result))
        # print('ng)
        # print(features,'\n', boxes,'\n', detect_result)
        # np.fromstring(a[1],dtype=np.int)
        # np.fromstring(a[2],dtype=np.float32)

        for res in detect_result:
            res['box'] = str(res['box'].tolist())
            res['score'] = str(res['score'])
            res['class_id'] = str(res['class_id'])
        # print(detect_result.tolist())
        # print(detect_result.tolist())
        # print(type(detect_result.tolist()))
        records = [(boxes.tostring(), features.tostring(), json.dumps(detect_result.tolist()), timestring), ]

        cur.executemany('INSERT INTO pedestrian(boxes, features,detect_result,timestamp) VALUES(?,?,?,?);', records);

        # print(f'We have inserted {cur.rowcount} records to the table, and delete file {filename}.')

        # commit the changes to db
        conn.commit()
        os.remove(filename)

        # break


def create_table(cur):
    try:
        cur.execute("create table pedestrian "
                    "(" \
                    "id            INTEGER not null" \
                    "primary key autoincrement," \
                    "boxes         TEXT    not null," \
                    "features      TEXT    not null," \
                    "detect_result TEXT," \
                    "timestamp     TEXT    not null)"
                    )
    except:
        pass


if __name__ == '__main__':
    with closing(sqlite3.connect("aquarium.sqlite")) as connection:
        with closing(connection.cursor()) as cursor:
            # cursor.execute("delete from pedestrian;")
            # connection.commit()
            create_table(cursor)
            while 1:
                read_npz_to_db(connection, cursor)
                time.sleep(5)
                # break
