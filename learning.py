import cv2
import numpy as np

#-------------------------------------------MEASUREMENT OF OYSTER IN CM---------------------------------------------

cap = cv2.VideoCapture(0) #real time video from webcam

while True:

    _, frame = cap.read()

    frame = frame[0:350, 15:540]

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    _, threshold = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)

    kernel = np.ones((5, 5), np.uint8)

    dilatation = cv2.dilate(threshold, kernel, iterations=1)

    countours, _ = cv2.findContours(dilatation, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for (i,c) in enumerate(countours):

        (x,y,w,h) = cv2.boundingRect(c)
        area = cv2.contourArea(c)
        cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 2)
        cm_x = (w*10)/71
        cm_y = (h*15)/120
        cv2.putText(frame, str(int(cm_x)) + " cm", (x+y+h+15, y+h+15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,255,255), 1)
        cv2.putText(frame, str(int(cm_y)) + " cm", (x, y+h+15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,255,255), 1)

    cv2.imshow("frame", frame)
    cv2.imshow("gray", gray) 
    cv2.imshow("threshold", threshold)
    cv2.imshow("dilatation", dilatation)
    cv2.imshow("countours", frame)

    key = cv2.waitKey(1)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()

#--------------------------------------------COUNTING THE NUMBER OF OYSTERS----------------------------------------------

#FUNÇÃO DO PONTO CENTRAL DO OBJETO
def center(x,y,w,h):
    x1 = int(w/2)
    y1 = int(h/2)
    cx = x+x1
    cy = y+y1

    return cx,cy

#CÂMERA OU VÍDEO
cap = cv2.VideoCapture('video.mp4') #video from the file
#cap = cv2.VideoCapture(0) #real time video from webcam

fgbg = cv2.createBackgroundSubtractorMOG2()

#VARIÁVEIS
posL = 150 #ajustar conforme a posição
offset = 30

xy1 = (20, posL)
xy2 = (300, posL)

detects = []

total = 0

#PROCESSAMENTO DE IMAGEM
while 1:

    ret, frame = cap.read()
    #_, frame = cap.read() #para o vídeo em tempo real, caso queira usar o vídeo do arquivo, descomente essa linha e comente a linha acima

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    fgmask = fgbg.apply(gray) #máscara: diferença entre o plano de fundo e o objeto em movimento (ou entre o frame anterior e o próximo frame)

    retval, threshold = cv2.threshold(fgmask, 200, 255, cv2.THRESH_BINARY) #elimina o ruído da máscara, deixando apenas os objetos em movimento
    #_, threshold = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY) #threshold para o vídeo em tempo real

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)) #kernel elíptico: melhor para objetos circulares

    opening = cv2.morphologyEx(threshold, cv2.MORPH_OPEN, kernel, iterations=2) #remove o ruído da imagem, deixando apenas os objetos em movimento mais definidos
    #ajustar iterations de acordo com a câmera
    dilatation = cv2.dilate(opening, kernel, iterations=8) #aumenta o tamanho dos objetos em movimento, preenchendo os buracos e unindo os objetos próximos, facilitando a contagem
    #ajustar iterations de acordo com o tamamho desejado do objeto

    closing = cv2.morphologyEx(dilatation, cv2.MORPH_CLOSE, kernel, iterations=8) #fecha os buracos dentro dos objetos, unindo os objetos próximos e facilitando a contagem

    countours, hierarchy = cv2.findContours(closing, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    #DESENHO DAS LINHAS NA IMAGEM
    cv2.line(frame, xy1, xy2, (255,0,0), 3) 
    cv2.line(frame, (xy1[0], posL-offset), (xy2[0], posL-offset), (255,255,0), 2)
    cv2.line(frame, (xy1[0], posL+offset), (xy2[0], posL+offset), (255,255,0), 2)

    i = 0
    for cnt in countours:

        (x,y,w,h) = cv2.boundingRect(cnt)
        area = cv2.contourArea(cnt)

        if int(area) > 3000: #ajustar o valor de acordo com o tamanho do objeto a ser contado, para eliminar objetos pequenos que não são ostras
            centro = center(x,y,w,h)
 
            cv2.putText(frame, str(i), (x+5, y+15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0),255,255), 2)
            cv2.circle(frame, centro, 4, (0,0,255), -1)
            cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 2)

            if len(detects) <= 1:
                detects.append([])

            if centro[1] > (posL-offset) and centro[1] < (posL+offset):
                detects[i].append(centro)
            else:
                detects[i].clear()

            i += 1

    if len(countours) == 0:
        detects.clear()
    else:
        for detect in detects:
            for (c,l) in enumerate(detect):
                #verificar se faz sentido essa lógica
                if detect[c-1][1] < posL and l[1] > posL:
                    detect.clear()
                    total += 1
                    cv2.line(frame, xy1, xy2, (0,0,255), 5)

                if c > 0:
                    cv2.line(frame, detect[c-1], detect[c], (0,0,255), 2) #linha de tracking do objeto

    print(centro)
    print(detects)

    cv2.putText(frame, "Total: " + str(total), (10,20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 2)

    cv2.imshow("frame", frame)
    cv2.imshow("gray", gray)
    cv2.imshow("fgmask", fgmask)
    cv2.imshow("threshold", threshold)
    cv2.imshow("opening", opening)
    cv2.imshow("dilatation", dilatation)
    cv2.imshow("closing", closing)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()