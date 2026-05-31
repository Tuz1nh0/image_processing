import cv2
import numpy as np

cap = cv2.VideoCapture('test.mp4')

fgbg = cv2.createBackgroundSubtractorMOG2(detectShadows=False)

windows = ["frame", "fgmask", "threshold", "opening", "dilatation", "closing"]

for name in windows:
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)  # Permite redimensionar
    cv2.resizeWindow(name, 500, 400)

while True:
    ret, frame = cap.read()

    if not ret:
        print("Fim do vídeo ou erro na leitura.")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    fgmask = fgbg.apply(gray)

    retval, threshold = cv2.threshold(fgmask, 10, 255, cv2.THRESH_BINARY)

    small_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

    opening = cv2.morphologyEx(threshold, cv2.MORPH_OPEN, small_kernel, iterations=1)

    large_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 30))

    dilatation = cv2.dilate(opening, large_kernel, iterations=2)

    closing = cv2.morphologyEx(dilatation, cv2.MORPH_CLOSE, large_kernel, iterations=2)

    countours, hierarchy = cv2.findContours(closing, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for (i,c) in enumerate(countours):
        (x,y,w,h) = cv2.boundingRect(c)
        area = cv2.contourArea(c)
        if int(area) < 500:
            continue
        cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 5)
        cm_x = (w*10)/71
        cm_y = (h*15)/120
        cv2.putText(frame, str(int(cm_x)) + " cm", (x+y+h+15, y+h+15), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 5)
        cv2.putText(frame, str(int(cm_y)) + " cm", (x, y+h+15), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 5)

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