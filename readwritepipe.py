import cv2
import os
import ffmpeg

FIFO = '/dev/shm/rtmp_q'
w = 640
h = 480
c = 3
# ffmpegprocess = (
#     ffmpeg.input('pipe:',
#                  # r='14',
#                  # hwaccel ='vdpau',
#                  hwaccel='vaapi',
#                  hwaccel_device='/dev/dri/card0')
#         .output('rtmp://192.168.8.121/live/bbb',
#                 # vcodec='libx264', pix_fmt='yuv420p', preset='veryfast',
#                 # r='14', g='50', video_bitrate='1.4M', maxrate='2M', bufsize='2M', segment_time='6',
#                 format='flv',
#                 # **{'c:v':'h264_vaapi'},
#                 # hwaccel_device='/dev/dri/card0'
#
#                 ).global_args('-c:v', 'h264_vaapi').run_async(pipe_stdin=True)
#
# )
# print(ffmpegprocess.compile())
cap = cv2.VideoCapture(0)
with open('/dev/shm/rtmp_q', 'wb') as fid:
    while 1:
        f, imgx = cap.read()
        if not f:
            continue
        # ret2, frame2 = cv2.imencode('.png', imgx)
        # fid.write(frame2)
        fid.write(imgx.tobytes())
        # ffmpegprocess.stdin.write(frame2.tobytes())
