import os
import uuid
import io
import cv2
import os
import numpy as np

FIFO = '/dev/shm/img_q'
w = 640
h = 480
c = 3
def gen_camera_frame(_):
    if not os.path.exists(FIFO):
        os.mkfifo(FIFO)
    with open(FIFO, 'rb') as fifo:
        print("FIFO opened")
        while True:
            # image_stream = io.BytesIO()
            data = fifo.read(w * h * c)
            filename = f'{uuid.uuid4().hex}'
            # img=cv2.imdecode(data, 1)
            # cv2.imwrite(f'/dev/shm/{uuid.uuid4().hex}.jpg',img)
            if len(data):
                # filename=f'/dev/shm/{uuid.uuid4().hex}'
                # with open(filename, 'wb') as fid:
                #     fid.write(data)

                # img=cv2.imdecode(data,1)
                # img=bts_to_img(data)
                # buff = np.fromstring(data, np.uint8)
                buff=np.frombuffer(data,np.uint8)
                img = cv2.imdecode(buff, cv2.IMREAD_COLOR)

                cv2.imwrite(f'/dev/shm/{filename}.jpg', img)
            else:
                print(f'data len is {len(data)}')
            print(f'read {filename}')
            # print('Read: "{0}"'.format(data))
