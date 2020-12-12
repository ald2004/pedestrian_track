import cv2
import os

for p in os.listdir('/dev/shm'):
    if len(p)>30:
        try:
            img = cv2.imread(f'/dev/shm/{p}')
            cv2.imwrite(f'/dev/shm/{p}.jpg', img)
        except:
            try:
                os.remove(f'/dev/shm/{p}')
            except:
                pass