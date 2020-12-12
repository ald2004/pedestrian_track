import cv2
import os

cap=cv2.VideoCapture(0)
f,i=cap.read()
while 1:
    if not f:
        continue
    with open('/dev/shm/rtmp_q','wb') as fifo:
        yuvimg=cv2.cvtColor(i, cv2.COLOR_RGB2YUV)
        yuvb=yuvimg.tobytes()

        xx=fifo.write(yuvb)
        print(len(yuvimg), '===============', len(yuvb), '=============', yuvimg.shape,'==========',xx)
        # print('ssssssss')