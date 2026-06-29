import cv2
import math

paused = False

ratio1 = 3 / 111
ratio2 = 4.5 / 169

points = []

def draw_circle(event,x,y,flags,param):
    global points
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(points) == 2:
            points = []
        points.append((x,y))
'''
windows = ["frame", "frame_display"]
for name in windows:
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(name, 700, 500)
'''
cv2.namedWindow("frame")
cv2.setMouseCallback("frame", draw_circle)

cap = cv2.VideoCapture('medição2.mp4')

ret, frame = cap.read()

while True:

    #height, width, _ = frame.shape
    #roi = frame[500: 1500, 200: 800]

    if not ret:
        print("Fim do vídeo ou erro na leitura.")
        break

    if not paused:
        ret, next_frame = cap.read()
        if ret:
            frame = next_frame

    frame_display = frame.copy()

    for pt in points:
        cv2.circle(frame, pt, 5, (0,255,255), -1)

    if len(points) == 2:
        pt1 = points[0]
        pt2 = points[1]
        distance_px = math.hypot(pt2[0] - pt1[0], pt2[1] - pt1[1])
        
        distance_cm = ratio * distance_px
        cv2.putText(frame_display, fr"{int(distance_px)}cm", (pt1[0], pt1[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.imshow("frame", frame_display)

    key = cv2.waitKey(30) & 0xFF
 
    if key == ord('q'):
        break
    elif key == 32:
        paused = not paused

cap.release()
cv2.destroyAllWindows()
    
