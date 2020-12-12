import os
import uuid
import io
import cv2
import os
import numpy as np
import ffmpeg

FIFO = '/dev/shm/rtmp_q'
FIFO = '/dev/shm/img_q'
w = 640
h = 480
c = 3

w = 320
h = 240
# ffmpegprocess = (
#     ffmpeg
#         .input('pipe:', r='6', hwaccel='vdpau', hwaccel_device='/dev/dri/card0')
#         .output('rtmp://192.168.8.121/live/bbb', vcodec='libx264', pix_fmt='yuv420p', preset='veryfast',
#                 r='20', g='50', video_bitrate='1.4M', maxrate='2M', bufsize='2M', segment_time='6',
#                 format='flv',
#                 # **{'c:v': 'h264_rkmpp'}
#                 )
#         .run_async(pipe_stdin=True))
# if not os.path.exists(FIFO):
#     os.mkfifo(FIFO)
with open(FIFO, 'rb') as fifo:
    print("FIFO opened")
    while True:
        # image_stream = io.BytesIO()
        data = fifo.read(w * h * c)
        if not len(data):
            continue

        buff = np.frombuffer(data, np.uint8)
        imgx = cv2.imdecode(buff, cv2.IMREAD_COLOR)
        cv2.imwrite(f'/dev/shm/{uuid.uuid4().hex}.jpg',imgx)
        # if not imgx:
        #     continue
        # ret2, frame2 = cv2.imencode('.png', imgx)
        # ffmpegprocess.stdin.write(data.tobytes())
        # filename = f'{uuid.uuid4().hex}'
        # img=cv2.imdecode(data, 1)
        # cv2.imwrite(f'/dev/shm/{uuid.uuid4().hex}.jpg',img)
        # if len(data):
            # filename=f'/dev/shm/{uuid.uuid4().hex}'
            # with open(filename, 'wb') as fid:
            #     fid.write(data)

            # img=cv2.imdecode(data,1)
            # img=bts_to_img(data)
            # buff = np.fromstring(data, np.uint8)
        #     buff=np.frombuffer(data.np.uint8)
        #     img = cv2.imdecode(buff, cv2.IMREAD_COLOR)
        #
        #     cv2.imwrite(f'/dev/shm/{filename}.jpg', img)
        # else:
        #     print(f'data len is {len(data)}')
        # print(f'read {filename}')
        # print('Read: "{0}"'.format(data))
