import datetime
import time
import csv
from detectFace import funcDetectFace
from detectPose import funcDetectPose
import cv2

def findStudentName(cap):
    old = funcDetectFace(cap)
    i=3
    while i>0:
        print(funcDetectFace(cap))
        new = funcDetectFace(cap)
        if(old == new):
            i=i-1
        else:
            print('face lost/changed -> resetting')
            i=5
            old = new
        time.sleep(1)
    return old

def findAngles(cap):
    #Find Angles for ctr seconds
    angles=[]
    i=5
    while i>0:
        ret=[]
        ret.append(funcDetectPose(cap))
        if ret == -1:
            print("no face found")
        elif ret == -2:
            print("camera not connected")
        ret.append(datetime.datetime.now())
        angles.append(ret)
        print(ret)
        i=i-1
        time.sleep(1)
    return angles

if __name__=='__main__':
    ctr = 5
    cap = cv2.VideoCapture(0)
    studentName = findStudentName(cap)
    #studentName=("aditya")
    angles = findAngles(cap)
    #angles = [(-2.9541107103044504, -0.80001947719528999, 0.72367030867424731), (-6.1926197064548782, 3.3914647826761044, -8.6323708765483271), (14.800405428511139, 4.9228644640433625, 6.8461287292403661), (20.707360555748618, 1.7134136849390902, 9.1795747334311315)]
    cv2.destroyAllWindows()
    cap.release()
    row = [studentName]
    for i in angles:
        row.append(i)
    print(row)
    with open('student.csv','a') as csvFile:
        writer = csv.writer(csvFile)
        writer.writerow(row)
