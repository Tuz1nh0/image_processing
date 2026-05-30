import cv2
import numpy as np

#-------------------------------------------MEASUREMENT OF OYSTER IN CM---------------------------------------------

cap = cv2.VideoCapture('test.mp4')

windows = ["frame", "threshold", "opening", "dilatation"]

for name in windows:
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)  # Permite redimensionar
    cv2.resizeWindow(name, 500, 400)

while True:

    ret, frame = cap.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    retval, threshold = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)

    kernel = np.ones((5, 5), np.uint8)

    dilatation = cv2.dilate(threshold, kernel, iterations=1)

    countours, _ = cv2.findContours(dilatation, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for (i,c) in enumerate(countours):

        (x,y,w,h) = cv2.boundingRect(c)
        area = cv2.contourArea(c)
        cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 2)
        cm_x = (w*10)/71
        cm_y = (h*15)/120
        cv2.putText(frame, str(int(cm_x)) + " cm", (x+y+h+15, y+h+15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,255,255), 1)
        cv2.putText(frame, str(int(cm_y)) + " cm", (x, y+h+15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,255,255), 1)

    cv2.imshow("frame", frame)
    #cv2.imshow("gray", gray) 
    cv2.imshow("threshold", threshold)
    cv2.imshow("dilatation", dilatation)
    cv2.imshow("countours", frame)

    key = cv2.waitKey(1)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()