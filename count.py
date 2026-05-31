import cv2
import numpy as np

def center(x,y,w,h):
    x1 = int(w/2)
    y1 = int(h/2)
    cx = x+x1
    cy = y+y1
    return cx,cy

cap = cv2.VideoCapture('test.mp4')

fgbg = cv2.createBackgroundSubtractorMOG2(detectShadows=False)

windows = ["frame", "fgmask", "threshold", "opening", "dilatation", "closing"]

for name in windows:
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)  # Permite redimensionar
    cv2.resizeWindow(name, 500, 400)

posL = 1050
offset = 100

xy1 = (posL, 400)
xy2 = (posL, 850)

detects = []

total = 0

while 1:
    ret, frame = cap.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    fgmask = fgbg.apply(gray)

    retval, threshold = cv2.threshold(fgmask, 10, 255, cv2.THRESH_BINARY)

    small_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

    opening = cv2.morphologyEx(threshold, cv2.MORPH_OPEN, small_kernel, iterations=1)

    large_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 30))

    dilatation = cv2.dilate(opening, large_kernel, iterations=2)

    closing = cv2.morphologyEx(dilatation, cv2.MORPH_CLOSE, large_kernel, iterations=2)

    countours, hierarchy = cv2.findContours(closing, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    cv2.line(frame, xy1, xy2, (255,0,0), 10)

    cv2.line(frame, (posL-offset,xy1[1]), (posL-offset,xy2[1]), (255,255,0), 10)

    cv2.line(frame, (posL+offset,xy1[1]), (posL+offset,xy2[1]), (255,255,0), 10)

    i = 0

    for cnt in countours:
        (x,y,w,h) = cv2.boundingRect(cnt)
        area = cv2.contourArea(cnt)
        if int(area) > 10000:
            centro = center(x,y,w,h)

            cv2.putText(frame, str(i), (x+5, y+50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,255,255), 2)
            cv2.circle(frame, centro, 10, (0,0,255), -1)
            cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 10)

            if len(detects) <= i:
                detects.append([])
            
            if centro[0] > posL-offset and centro [0] < posL+offset:
                detects[i].append(centro)
            else:
                detects[i].clear()

            i+=1

    if len(countours) == 0:
        detects.clear()

    for detect in detects:
        for (count,line) in enumerate(detect):
            if detect[count-1][0] < posL and line[0] > posL:
                detects.clear()
                total+=1
                cv2.line(frame, xy1, xy2, (0,0,255), 5)
                continue

            if count > 0:
                cv2.line(frame, detect[count-1], line, (0,0,255), 5)

    cv2.putText(frame, "TOTAL: "+str(total), (150,200), cv2.FONT_HERSHEY_SIMPLEX, 3, (0,255,255), 2)

    cv2.imshow("frame", frame)
    #cv2.imshow("gray", gray)
    cv2.imshow("fgmask", fgmask)
    cv2.imshow("threshold", threshold)
    cv2.imshow("opening", opening)
    cv2.imshow("dilatation", dilatation)
    cv2.imshow("closing", closing)

    if cv2.waitKey(30) & 0xFF == ord('q'):
        break
    
cap.release()
cv2.destroyAllWindows()