from ctypes import *
import threading
import numpy as np
import datetime
import os
import cv2
import copy
import ctypes
import multiprocessing
import time
# cur = cdll.LoadLibrary('./libEncode.so')


def xxfun():
    cur = CDLL(os.path.join(os.environ.get('ESW_PATH', './'), "libEncode.so"), RTLD_GLOBAL)

    class dataInPack(Structure):
        _fields_ = [("buf", c_ubyte * 460800)]  # buf1

    class dataOutPack(Structure):
        _fields_ = [("len", c_int),  # width
                    ("buf1", c_ubyte * 2000000)]  # buf1

    cur.readH264.restype = (dataOutPack)

    inData = dataInPack()

    cur.create_encoder(640, 480, 10)
    # print('qqqqqqqqqqqqqqq')
    frame = cv2.imread('/opt/pedestrian_track/images/boe_test.jpg')
    frame = cv2.resize(frame, (640, 480))
    # # print(frame)
    # print(frame.shape)
    yuv = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV_I420)
    # print('rrrrrrrrrrrrrrrr')
    # print(yuv.shape)
    frame_for_encoder = copy.deepcopy(yuv)
    # print(frame_for_encoder)
    # print('wwwwwwwwwwwwww')
    inData.buf = (ctypes.c_ubyte * 460800).from_buffer_copy(frame_for_encoder)
    print(inData.buf)
    print('eeeeeeeeeeeeeeeee')
    cur.inputBgr(inData)
    print('etttttttttttttttttttttt')
    img = cur.readH264()

    print('yyyyyyyyyyyyyyyy')
    print(img.len)
    print(img.buf1[:img.len])
    # ffmpegprocess.stdin.write(bytes(img.buf1[:img.len]))
    # print('zzzzzzzzzzzz')


if __name__=='__main__':
    pp=multiprocessing.Process(target=xxfun,args=())
    pp.start()
    while 1:
        time.sleep(99)