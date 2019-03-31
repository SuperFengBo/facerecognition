#coding = utf-8
import cv2
import os
import time

face_cascade = cv2.CascadeClassifier(r'./haarcascade_frontalface_default.xml')
n=1
while(n < 23):
    readpath = os.path.join('input_jimg/' + str(n) + '.tiff')
    print('readpath is ',readpath)
    frame = cv2.imread(readpath)
    cv2.imshow('frame',frame)    
#    ret,frame = cap.read()
#    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

#facedetect
    faces = face_cascade.detectMultiScale(
    frame,
    scaleFactor = 1.15,
    minNeighbors = 5,
    minSize = (5,5),
    # flags = cv2.cv.CV_HAAR_SCALE_IMAGE
    )
    print('\n\nlen(faces) is',len(faces),faces)
    print ("发现{0}个人脸!".format(len(faces)))

    for(x,y,w,h) in faces:
#crop
        #cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
        print ('frame x is %d y is %d h is %d w is %d'%(x,y,h,w))           
        corpimg = frame[y :y+h ,x :x+w]
        print ('corpimg x is %d y is %d h is %d w is %d'%(x,y,h,w)) 

#resize
        print ('corpimg.shape',corpimg.shape)
        r = 64.0/corpimg.shape[1]
        dim = (64,int(corpimg.shape[0]*r))
        resizeimg = cv2.resize(corpimg,dim,interpolation=cv2.INTER_AREA)

        cv2.imshow('corp',resizeimg)
        imgpath = os.path.join('./mingxing_faces/' + str(n)+'.jpg')
        print('imgpath is',imgpath)
        cv2.imwrite(imgpath,resizeimg)
        n = n + 1
#        time.sleep(0.5)
    #cv2.imshow('cam',frame)
    if cv2.waitKey(1)==ord('f'):
        break




