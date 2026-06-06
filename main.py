import cv2
import numpy as np

def center(x,y,w,h):
    x1 = int(w/2)
    y1 = int(h/2)
    cx = x+x1
    cy = y+y1
    return cx,cy

cap = cv2.VideoCapture('teste_final_1.mp4')

fgbg = cv2.createBackgroundSubtractorMOG2()

windows = ["frame", "fgmask", "threshold", "opening", "dilatation"]
for name in windows:
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)  # Permite redimensionar
    cv2.resizeWindow(name, 500, 400)

posL = 1050
offset = 100

xy1 = (posL, 400)
xy2 = (posL, 850)

detects = []

total = 0

while True:
    ret, frame = cap.read()

    if not ret:
        print("Fim do vídeo ou erro na leitura.")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    fgmask = fgbg.apply(gray)

    retval, threshold = cv2.threshold(fgmask, 100, 255, cv2.THRESH_BINARY)

    kernel = np.ones((5,5), np.uint8)

    opening = cv2.morphologyEx(threshold, cv2.MORPH_OPEN, kernel, iterations=1)

    dilatation = cv2.dilate(threshold, kernel, iterations=1)

    countours, hierarchy = cv2.findContours(dilatation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for (i,c)  in enumerate(countours):
        area = cv2.contourArea(c)
        if int(area) > 5000:
            (x,y,w,h) = cv2.boundingRect(c)
            centro = center(x,y,w,h)
            cv2.circle(frame, centro, 10, (0,0,255), -1)

            cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)
        
    cv2.imshow("frame", frame)
    #cv2.imshow("gray", gray)
    cv2.imshow("opening", opening)
    cv2.imshow("fgmask", fgmask)
    cv2.imshow("threshold", threshold)
    cv2.imshow("dilatation", dilatation)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
    