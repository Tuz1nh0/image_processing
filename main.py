import cv2
import numpy as np

def center(x,y,w,h):
    x1 = int(w/2)
    y1 = int(h/2)
    cx = x+x1
    cy = y+y1
    return cx,cy

cap = cv2.VideoCapture('teste_final.mp4')

fgbg = cv2.createBackgroundSubtractorMOG2(detectShadows=True)

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

while True:
    ret, frame = cap.read()

    if not ret:
        print("Fim do vídeo ou erro na leitura.")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    fgmask = fgbg.apply(gray)

    retval, threshold = cv2.threshold(fgmask, 250, 255, cv2.THRESH_BINARY)

    #kernel = np.ones((5,5), np.uint8)
    #small_kernel = np.ones((3,3), np.uint8)
    #large_kernel = np.ones((7,7), np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    #small_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    #large_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 30))

    opening = cv2.morphologyEx(threshold, cv2.MORPH_OPEN, kernel, iterations=1)

    dilatation = cv2.dilate(threshold, kernel, iterations=2)

    closing = cv2.morphologyEx(dilatation, cv2.MORPH_CLOSE, kernel, iterations=1)

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
    cv2.imshow("closing", closing)

    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
    