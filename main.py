import cv2
import numpy as np
import math

paused = False

def center(x,y,w,h):
    x1 = int(w/2)
    y1 = int(h/2)
    cx = x+x1
    cy = y+y1
    return cx,cy

cap = cv2.VideoCapture('teste_final.mp4')

fgbg = cv2.createBackgroundSubtractorMOG2(detectShadows=True)

windows = ["frame", "roi", "threshold", "opening", "dilatation", "closing"]
for name in windows:
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(name, 700, 500)

ratio = 3 / 111

posL = 1400
offset = 100

xy1 = (200, posL)
xy2 = (800, posL)

detects = []

total = 0

ret, frame = cap.read()

while True:

    if not ret:
        print("Fim do vídeo ou erro na leitura.")
        break

    if not paused:
        ret, next_frame = cap.read()
        if ret:
            frame = next_frame

    frame_display = frame.copy()

    height, width, _ = frame_display.shape
    roi = frame_display[500: 1500, 200: 800]

#-----------------------------------------------------------------------------------------------------------------------------

    gray = cv2.cvtColor(frame_display, cv2.COLOR_BGR2GRAY)

    fgmask = fgbg.apply(gray)

    retval, threshold = cv2.threshold(fgmask, 250, 255, cv2.THRESH_BINARY)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    large_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))

    opening = cv2.morphologyEx(threshold, cv2.MORPH_OPEN, kernel, iterations=1)

    dilatation = cv2.dilate(threshold, kernel, iterations=2)

    closing = cv2.morphologyEx(dilatation, cv2.MORPH_CLOSE, kernel, iterations=2)

    countours, hierarchy = cv2.findContours(dilatation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

#--------------------------------------------------------------------------------------------------------------------------------

    cv2.line(frame_display, xy1, xy2, (255,0,255), 4)
    cv2.line(frame_display, (xy1[0], posL-offset), (xy2[0], posL-offset), (255, 255, 0), 4)
    cv2.line(frame_display, (xy1[0], posL+offset), (xy2[0], posL+offset), (255, 255, 0), 4)

    i = 0
    for cnt in countours:
        area = cv2.contourArea(cnt)
        if int(area) > 6500:
            (x,y,w,h) = cv2.boundingRect(cnt)
            centro = center(x,y,w,h)

            cv2.putText(frame_display, str(i), (x+5, y+30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 4)
            cv2.circle(frame_display, centro, 10, (0,0,255), -1)
            cv2.rectangle(frame_display, (x,y), (x+w,y+h), (0,255,0), 3)
            
            cm_x = (w * 3) / 111
            cm_y = (h * 4.5) / 169 
            cv2.putText(frame_display, fr"{int(cm_x)}cm", (x, y+h+30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,0,0), 3) 
            cv2.putText(frame_display, fr"{int(cm_y)}cm", (x+w+5, y+5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,0,0), 3)
           
            
            if len(detects) <= i:
                detects.append([])

            if centro[1] > posL-offset and centro[1] < posL+offset:
                detects[i].append(centro)
            else:
                detects[i].clear()

            i += 1

    if len(countours) == 0:
        detects.clear()
    else:
        for detect in detects:
            for (c,l) in enumerate(detect):
                if detect[c-1][1] > posL and l[1] < posL:
                    detect.clear()
                    total += 1
                    cv2.line(frame_display, xy1, xy2, (0,255,0), 5)
                    continue

                if c > 0:
                    cv2.line(frame_display, detect[c-1], l, (0,0,255), 1) 

    cv2.putText(frame_display, "TOTAL: "+str(total), (60,90), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,255,255), 3)
    
    cv2.imshow("frame", frame_display)
    cv2.imshow("roi", roi)
    #cv2.imshow("gray", gray)
    cv2.imshow("opening", opening)
    #cv2.imshow("fgmask", fgmask)
    cv2.imshow("threshold", threshold)
    cv2.imshow("dilatation", dilatation)
    cv2.imshow("closing", closing)

    key = cv2.waitKey(30) & 0xFF

    if key == ord('q'):
        break
    elif key == 32:
        paused = not paused

cap.release()
cv2.destroyAllWindows()
    
