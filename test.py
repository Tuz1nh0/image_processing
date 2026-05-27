#código feito com ajuda do gemini e chatgpt

import cv2
import numpy as np

def center(x,y,w,h):
    x1 = int(w/2)
    y1 = int(h/2)
    cx = x+x1
    cy = y+y1

    return cx,cy

#cap = cv2.VideoCapture('video.mp4')
cap = cv2.VideoCapture(0)

fgbg = cv2.createBackgroundSubtractorMOG2()

posL = 320
offset = 30

#linha vertical
y_topo = 10
y_fundo = 350
xy1 = (posL, y_topo)
xy2 = (posL, y_fundo)

tracked = []

track_id = 0
total = 0

while 1:
    _, frame = cap.read()
    #ret, frame = cap.read('video.mp4') #para o vídeo pré gravado
    
    #frame = frame[0:480, 0:640]

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    fgmask = fgbg.apply(gray) 

    _, threshold = cv2.threshold(fgmask, 200, 255, cv2.THRESH_BINARY) 
    #retval, threshold = cv2.threshold(fgmask, 200, 255, cv2.THRESH_BINARY)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)) 

    opening = cv2.morphologyEx(threshold, cv2.MORPH_OPEN, kernel, iterations=2) 

    dilatation = cv2.dilate(opening, kernel, iterations=8) 

    closing = cv2.morphologyEx(dilatation, cv2.MORPH_CLOSE, kernel, iterations=8) 

    countours, _ = cv2.findContours(closing, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #countours, hierarchy = cv2.findContours(closing, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    cv2.line(frame, xy1, xy2, (255,0,0), 3) 
    cv2.line(frame, (posL-offset, y_topo), (posL-offset, y_fundo), (255,255,0), 2)
    cv2.line(frame, (posL+offset, y_topo), (posL+offset, y_fundo), (255,255,0), 2)

    i = 0
    for cnt in countours:
        (x,y,w,h) = cv2.boundingRect(cnt)
        area = cv2.contourArea(cnt)

        if area > 3000:

            #LOGICA DA MEDIÇÃO DO OBJETO
            cm_x = (w * 10) / 71
            cm_y = (h * 15) / 120
            #cv2.putText(frame, str(int(cm_x)) + " cm", (x+y+h+15, y+h+15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,255,255), 1)
            #cv2.putText(frame, str(int(cm_y)) + " cm", (x, y+h+15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,255,255), 1)
            cv2.putText(frame, f"W: {int(cm_x)} cm", (x, y - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
            cv2.putText(frame, f"H: {int(cm_y)} cm", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)

            centro = center(x,y,w,h)
            #cx, cy = center(x,y,w,h)

            cv2.circle(frame, (centro), 5, (0,0,255), -1)
            #cv2.circle(frame, (cx,cy), 5, (0,0,255), -1)
            cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 2)

            #LÓGICA DE CONTAGEM DO OBJETO
            if len(tracked) <= 1:
                tracked.append([])

            if centro[0] > (posL-offset) and centro[0] < (posL+offset):
                tracked[i].append(centro)
            else:
                tracked[i].clear()

            i += 1
    if len(countours) == 0:
        tracked.clear()
    else:
        for track in tracked:
            for (c,l) in enumerate(track):
                #verificar se faz sentido essa lógica
                if track[c-1][1] < posL and l[1] > posL:
                    track.clear()
                    total += 1
                    cv2.line(frame, xy1, xy2, (0,0,255), 5)

                if c > 0:
                    cv2.line(frame, track[c-1], track[c], (0,0,255), 2) #linha de tracking do objeto

    cv2.putText(frame, "Total: " + str(total), (10,20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 2)

    cv2.imshow("frame", frame)
    cv2.imshow("closing", closing)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()