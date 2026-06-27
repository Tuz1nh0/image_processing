import cv2
import math

points = []

def draw_circle(event,x,y,flags,param):
    global points
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(points) == 2:
            points = []
        points.append((x,y))

cv2.namedWindow("frame")
cv2.setMouseCallback("frame", draw_circle)

cap = cv2.VideoCapture('teste_final.mp4')

while True:
    ret, frame = cap.read()

    height, width, _ = frame.shape
    roi = frame[500: 1500, 200: 800]

    if not ret:
        print("Fim do vídeo ou erro na leitura.")
        break

    for pt in points:
        cv2.circle(frame, pt, 5, (0,255,255), -1)

    if len(points) == 2:
        pt1 = points[0]
        pt2 = points[1]
        distance = math.hypot(pt2[0] - pt1[0], pt2[1] - pt1[1])
        cv2.putText(frame, fr"{int(distance)}", (pt1[0], pt1[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.imshow("frame", frame)

    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
    
