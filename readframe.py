import cv2
import time

cap = cv2.VideoCapture(0)
if not (cap.isOpened()):
    print("Could not open video device")
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
while 1:
    ret, frame = cap.read()
    success, buffer = cv2.imencode(".jpg", frame)
    buffer.tofile('video_feed1')
    # with open('video_feed1', 'wb') as fid:
    #     cv2.imwrite(fid, frame)
    # cv2.imshow('preview',frame)
    # if cv2.waitKey(1) & 0xFF == ord('q'):
    #     break
    time.sleep(.5)
cap.release()
cv2.destroyAllWindows()
