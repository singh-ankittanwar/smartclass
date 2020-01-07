import numpy as np
import cv2
face_cascade = cv2.CascadeClassifier('D:\\Downloads\\opencv\\build\\etc\\haarcascades\\haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture(0)
rec = cv2.face.LBPHFaceRecognizer_create()
rec.read("C:\\Users\\adith\\Desktop\\minor\\face recognizer (ours)\\dataset\\yml\\trainingdata.yml")
id=0
font=cv2.FONT_HERSHEY_SIMPLEX
while True:
    ret, img = cap.read()
    ###img = cv2.imread("C:\\Users\\adith\\Desktop\\minor\\face recognizer (ours)\\dataset\\images\\1.1.jpg")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.5, 5)
    print(faces)
    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        id,conf=rec.predict(gray[y:y+h,x:x+w])
        if id == 1:
            id = "aditya"
        if id == 2:
            id = "chirag"
        if id == 3:
            id = "ankit"
        if id == 4:
            id = "rahul"
        else:
            id = "unknown"
        cv2.putText(img,str(id),(x+30,y+h+30),font,1,(255,255,255),2,cv2.LINE_AA)
        print(str(id))
        cv2.imshow('img',img)
    
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
