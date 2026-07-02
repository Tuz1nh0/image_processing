import cv2
import numpy as np
import math
import csv

paused = False

cap = cv2.VideoCapture('teste_final.mp4')

windows = ["frame", "roi"] #need to add the other windows
for name in windows:
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(name, 500, 350)

posL = 900
offset = 40
xy1 = (0, posL)
xy2 = (600, posL)

detects = []

total = 0

csv_file = open("oyster_data.csv", "w", newline="")
writer = csv.writer(csv_file)
writer.writerow(["Oyster Nº", "Width (cm)", "Height (cm)"])

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

#------------------------------------------IMAGE FILTERING AND MORPHOLOGIC OPERATIONS------------------------------------

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    blur = cv2.GaussianBlur(gray, (7, 7), 0)

    retval, threshold = cv2.threshold(blur, 130, 255, cv2.THRESH_BINARY_INV)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    opening = cv2.morphologyEx(threshold, cv2.MORPH_OPEN, kernel, iterations=1)

    dilatation = cv2.dilate(opening, kernel, iterations=2)

    closing = cv2.morphologyEx(dilatation, cv2.MORPH_CLOSE, kernel, iterations=5)

    contours, hierarchy = cv2.findContours(closing, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

#-------------------------------------------------COUNT AND MEASUREMENT LOGICS--------------------------------------------
    
    cv2.line(roi, xy1, xy2, (255,0,255), 4)
    cv2.line(roi, (xy1[0], posL-offset), (xy2[0], posL-offset), (255, 255, 0), 4)
    cv2.line(roi, (xy1[0], posL+offset), (xy2[0], posL+offset), (255, 255, 0), 4)
    
    i = 0
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if int(area) > 6500 and len(cnt) >= 5:
            ellipse = cv2.fitEllipse(cnt)
            (cx,cy), (w,h), angle = ellipse
            if w > 1000 or h > 1000:
                continue
            centro = (int(cx), int(cy))
            cv2.putText(roi, str(i), (int(cx)-10, int(cy)-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 4)
            cv2.circle(roi, centro, 10, (0,0,255), -1)
            cv2.ellipse(roi, ellipse, (0,255,0), 3)
            
            cm_x = float(w * 3) / 111
            cm_y = float(h * 4.5) / 169 
            cv2.putText(roi, f"{cm_x:.1f}cm", (int(cx - w/2), int(cy + h/2 + 30)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,0,0), 3) 
            cv2.putText(roi, f"{cm_y:.1f}cm", (int(cx + w/2 + 5), int(cy)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,0,0), 3)
            
            if len(detects) <= i:
                detects.append([])

            if centro[1] > posL-offset and centro[1] < posL+offset:
                detects[i].append({"center": centro, "width": cm_x, "height": cm_y})
            else:
                detects[i].clear()

            i += 1
    
    if len(contours) == 0:
        detects.clear()
    else:
        for detect in detects:
            for c,data in enumerate(detect):
                centro = data["center"]
                centro_ant = detect[c-1]["center"]
                if centro_ant[1] > posL and centro[1] < posL:
                    detect.clear()
                    total += 1
                    writer.writerow([total, round(data["width"], 2), round(data["height"], 2)])
                    csv_file.flush()
                    cv2.line(roi, xy1, xy2, (0,255,0), 5)
                    continue

                if c > 0:
                    cv2.line(roi, centro_ant, centro, (0,0,255), 1) 

    cv2.putText(roi, "TOTAL: "+str(total), (40,70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 3)
    
    cv2.imshow("frame", frame)
    #cv2.imshow("frame", frame_display)
    cv2.imshow("roi", roi)
    #cv2.imshow("gray", gray)
    #cv2.imshow("blur", blur)
    #cv2.imshow("threshold", threshold)
    #cv2.imshow("opening", opening)
    #cv2.imshow("dilatation", dilatation)
    #cv2.imshow("closing", closing)

    key = cv2.waitKey(30) & 0xFF

    if key == ord('q'):
        break
    elif key == 32:
        paused = not paused

csv_file.close()
cap.release()
cv2.destroyAllWindows()