import cv2
import numpy as np
import time
import os
import hand_tracking_module as htm

folerPath = "header"
mylist = os.listdir(folerPath)

overlaylist = []
for img in mylist:
    image = cv2.imread(f'{folerPath}/{img}')
    overlaylist.append(image)
header = overlaylist[0]
drawcolor = (0, 0, 0)

cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

detector = htm.handDetector()

brushthickness = 15
rubberthickness = 50
xp, yp = 0, 0
imgCanvas = np.zeros((720, 1280, 3), dtype='uint8')
# imgCanvas = cv2.resize(imgCanvas, (1280, 720)).astype(np.float32)
imgCanvas[:] = (0, 0, 0)

while True:
    suc, vid = cap.read()
    vid = cv2.flip(vid, 1)

    #hand landmarks
    vid = detector.findHands(vid)
    lmlist = detector.findPosition(vid, draw=True)

    if len(lmlist)!=0:

        #tips of index and middle finger
        x1, y1 = lmlist[8][1:]
        x2, y2 = lmlist[12][1:]
        #which fingers are up
        fingers = detector.fingersUp()
        #for 2 fingers up
        if fingers[1] and fingers[2]:
            xp, yp = 0, 0
            print("Selection Mode")
            if y1 < 125:
                if 250 < x1 < 450:
                    header = overlaylist[0]
                    drawcolor = (0, 255, 255)
                elif 550 < x1 < 750:
                    header = overlaylist[1]
                    drawcolor = (0, 169, 0)
                elif 800 < x1 < 950:
                    header = overlaylist[2]
                    drawcolor = (0, 0, 169)
                elif 1050 < x1 < 1200:
                    header = overlaylist[3]
                    drawcolor = (0, 0, 0)
            cv2.rectangle(vid, (x1, y1 - 35), (x2, y2 + 35), drawcolor, cv2.FILLED)

        #for index finger
        if fingers[1] and fingers[2]==False:
            cv2.circle(vid, (x1, y1), 15, drawcolor, cv2.FILLED)
            print("Drawing Mode")
            if xp==0 and yp==0:
                xp, yp = x1, y1
            if drawcolor == (0, 0, 0):
                cv2.line(vid, (xp, yp), (x1, y1), drawcolor, rubberthickness)
                cv2.line(imgCanvas, (xp, yp), (x1, y1), drawcolor, rubberthickness)
            else:
                cv2.line(vid, (xp, yp), (x1, y1), drawcolor, brushthickness)
                cv2.line(imgCanvas, (xp, yp), (x1, y1), drawcolor, brushthickness)

            xp, yp = x1, y1


    greyimg = cv2.cvtColor(imgCanvas, cv2.COLOR_BGR2GRAY)
    _, vidInv = cv2.threshold(greyimg, 50, 250, cv2.THRESH_BINARY_INV)
    vidInv = cv2.cvtColor(vidInv, cv2.COLOR_GRAY2BGR)

    vid = cv2.bitwise_and(vid, vidInv)
    vid = cv2.bitwise_or(vid, imgCanvas)

    vid[0:125, 0:1280] = header
    # vid = cv2.addWeighted(vid, 0.8, imgCanvas, 0.1, 0.0, dtype=cv2.CV_32F)


    cv2.imshow("VIDEO", vid)
    # cv2.imshow("CANVAS", imgCanvas)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
