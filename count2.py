#teste

import cv2
import numpy as np

cap = cv2.VideoCapture('test.mp4')

fgbg = cv2.createBackgroundSubtractorMOG2(detectShadows=False)

windows = ["frame", "gray", "fgmask", "threshold", "dilatation"]

for name in windows:
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)  # Permite redimensionar
    cv2.resizeWindow(name, 500, 400)

counter = 0
free = False

while True:
    ret, frame = cap.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    fgmask = fgbg.apply(gray)

    retval, threshold = cv2.threshold(fgmask, 10, 255, cv2.THRESH_BINARY)

    kernel = np.ones((8,8), np.uint8)

    dilatation = cv2.dilate(threshold, kernel, iterations=2)

    x,y,w,h = 1050, 400, 100, 500
    cut  = dilatation[y:y+h, x:x+w]
    white = cv2.countNonZero(cut)

    if white > 4000 and free == True:
        counter+=1
    
    if white < 4000:
        free = True
    else:
        free = False

    if free == False:
        cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 4)
    else:
        cv2.rectangle(frame, (x,y), (x+w, y+h), (255,0,255), 4)

    cv2.rectangle(threshold, (x,y), (x+w, y+h), (255,255,255), 6)

    cv2.putText(frame, str(white), (x-30,y-50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 5)
    cv2.putText(frame, "TOTAL: "+str(counter), (150,200), cv2.FONT_HERSHEY_SIMPLEX, 3, (255,0,0), 5)

    cv2.imshow("frame", frame)
    cv2.imshow("gray", gray)
    cv2.imshow("fgmask", fgmask)
    cv2.imshow("threshold", threshold)
    cv2.imshow("dilatation", dilatation)

    if cv2.waitKey(30) & 0xFF == ord('q'):
        break
    
cap.release()
cv2.destroyAllWindows()