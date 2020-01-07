import numpy as np
import cv2
from detectPose import funcDetectPose

def funcDetectFace(cap):
    face_cascade = cv2.CascadeClassifier('D:\\Downloads\\opencv\\build\\etc\\haarcascades\\haarcascade_frontalface_default.xml')
    rec = cv2.face.LBPHFaceRecognizer_create()
    rec.read("C:\\Users\\adith\\Desktop\\minor\\face recognizer (ours)\\dataset\\yml\\trainingdata.yml")
    id=0
    font=cv2.FONT_HERSHEY_SIMPLEX
    while True:
        ret, img = cap.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.5, 5)
        
        for (x,y,w,h) in faces:
            cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
            id,conf=rec.predict(gray[y:y+h,x:x+w])
            conf=100-float(conf)
            if id == 1:
                id = "aditya"
            elif id == 2:
                id = "chirag"
            elif id == 3:
                id = "ankit"
            elif id == 4:
                id = "rahul"
            else:
                id = "unknown"
            if (conf < 10):
                id = "unknown"
            cv2.putText(img,"Student name: "+str(id),(x+30,y+h+30),font,1,(255,255,255),2,cv2.LINE_AA)
            cv2.putText(img,"Confidence: "+str(conf)+"%",(x+30,y+h+50),font,1,(255,255,255),2,cv2.LINE_AA)
            #return str(id)

            crop_img = img[y-30:y+(2*h), x-30:x+(2*w)]
            cv2.imshow("cropped", crop_img)
            print(funcDetectPose(cap,crop_img))
            #cv2.imshow('img',img)
            
        
        if cv2.waitKey(1) == ord('q'):
            cv2.destroyAllWindows()
            break

if __name__ == '__main__':
    cap = cv2.VideoCapture(0)
    funcDetectFace(cap)
    cap.release()
