import numpy as np
import cv2
import time
import subprocess


cap1 = cv2.VideoCapture(4)
cap2 = cv2.VideoCapture(2)
cap1.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap1.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)
cap2.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap2.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)
# cnt = 0
delay = 1


while True:
    ret1, frame1 = cap1.read()
    ret2, frame2 = cap2.read()

    frame1 = frame1[11:, 15:]
    # frame2 = frame2[:339, :]
    frame1 = cv2.copyMakeBorder(frame1, 11, 0, 15, 0, cv2.BORDER_CONSTANT, 0)
    # frame2 = cv2.copyMakeBorder(frame2, 0, 21, 0, 0, cv2.BORDER_CONSTANT, 0)
    mat = cv2.getRotationMatrix2D((320, 180), -1.5, 1)
    frame1 = cv2.warpAffine(frame1, mat, (640, 360))

    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    hogParams = {'winStride': (8, 8), 'padding': (32, 32), 'scale': 1.05}
    human1, r1 = hog.detectMultiScale(frame1, **hogParams)
    human2, r2 = hog.detectMultiScale(frame2, **hogParams)

    point = []
    for(x, y, w, h) in human1:
        cv2.rectangle(frame1, (x, y), (x+w, y+h), (0, 0, 0), 3)
        point.append((x+w//2, y+h//2))

    for(x, y, w, h) in human2:
        cv2.rectangle(frame2, (x, y), (x+w, y+h), (0, 50, 255), 3)

    stereo = cv2.StereoBM_create(numDisparities=16, blockSize=15)
    frame11 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    frame22 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    frame11 = cv2.GaussianBlur(cv2.equalizeHist(frame11),(15,15), 7)
    frame22 = cv2.GaussianBlur(cv2.equalizeHist(frame22),(15,15), 7)
    out = stereo.compute(frame11, frame22)


    # cv2.imshow('frame1', frame1)
    # cv2.imshow('frame2', frame2)
    # print(out)
    print(np.count_nonzero(out == -16))
    # print('')
    out = np.where(out<0, 0, out)
    out = out.astype(np.uint8)

    sorted(point)
    for i in range(len(point) - 1):
        x0 = point[i][0]
        y0 = point[i][1]
        x1 = point[i+1][0]
        y1 = point[i+1][1]

        # if (abs(out[y1][x1] - out[y0][x0]) <= 60) and (abs(x1 - x0) <= 100) and (out[y1][x1] + out[y0][x0] != 0):
        if (abs(out[y1][x1] - out[y0][x0]) <= 60) and (abs(x1 - x0) <= 100):
            # print(cnt, "密です")
            subprocess.call("mpg321 mitsudesu.mp3", shell=True)
            cv2.circle(frame1, (x0, y0), 15, (0, 50, 255), thickness=-1)
            cv2.putText(frame1, '3Cs', (x0, y0), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), thickness=2)
            cv2.circle(frame1, (x1, y1), 15, (0, 50, 255), thickness=-1)
            cv2.putText(frame1, '3Cs', (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), thickness=2)
            # cnt += 1

    cv2.imshow('frame1', frame1)
    cv2.imshow('frame2', frame2)
    cv2.imshow('shisa', out)

    if cv2.waitKey(delay) & 0xFF == ord('q'):
        break
    # time.sleep(1)
# cv2.imwrite('./frame1.png', frame1)
# cv2.imwrite('./frame2.png', frame2)
# cv2.imwrite('./out.png', out)
