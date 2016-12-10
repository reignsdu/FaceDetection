#-*- coding: UTF-8 -*-
import cv2
import numpy as np
cv2.namedWindow("reigns")#命名一个窗口
cap=cv2.VideoCapture(0)#打开摄像头
success,frame=cap.read()#读取一桢图像，前一个返回值是是否成功，后一个返回值是图像本身
classifier=cv2.CascadeClassifier('/home/reigns/Public/opencv-3.1.0/data/haarcascades/haarcascade_frontalface_alt.xml')   #确保此xml文件与该py文件在一个文件夹下，或在open-cv安装包中找到
while success:
    success,frame=cap.read()
    size=frame.shape[:2]#获得当前桢彩色图像的大小
    image=np.zeros(size,dtype=np.float16)
    image=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    cv2.equalizeHist(image,image)
    divisor=8
    h,w=size
    minSize=(w/divisor,h/divisor)
    faceRects=classifier.detectMultiScale(image,1.2,2,cv2.CASCADE_SCALE_IMAGE,minSize)
    if len(faceRects)>0:
        for faceRect in faceRects:
            x,y,w,h=faceRect
            cv2.circle(frame,(x+w/2,y+h/2),min(w/2,h/2),(255,0,0))
            cv2.circle(frame,(x+w/3,y+5*h/13),min(w/8,h/8),(255,0,0))
            cv2.circle(frame,(x+2*w/3,y+5*h/13),min(w/8,h/8),(255,0,0))
            cv2.rectangle(frame,(x+3*w/8,y+3*h/4),(x+5*w/8,y+7*h/8),(255,0,0))
    cv2.imshow("reigns",frame) 
    key=cv2.waitKey(10)
    c=chr(key&255)
    if c in ['q','Q',chr(27)]:
        break
cv2.destroyWindow("reigns")
