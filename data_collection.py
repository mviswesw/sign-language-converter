#!/usr/bin/env python
# coding: utf-8

# In[63]:


import os
import cv2
import numpy as np


#------------------------------modified code begin--------------------------------------
if not os.path.exists("data"):
    os.makedirs("data")
    os.makedirs("data/train")
    os.makedirs("data/test")
    
    os.makedirs("data/train/a")
    os.makedirs("data/train/b")
    os.makedirs("data/train/c")
    os.makedirs("data/train/v")
   
    os.makedirs("data/test/a")
    os.makedirs("data/test/b")
    os.makedirs("data/test/c")
    os.makedirs("data/test/v")

# Train or test 
test_or_train = str(input("do you want to test or train? "))
directory = 'data/'+test_or_train+'/'

real_time_capture = cv2.VideoCapture(0)

while True:
    _, frame = real_time_capture.read()
   #Simulating mirror image
    frame = cv2.flip(frame, 1)
    
    add_up = {'a': len(os.listdir(directory+"/a")),
              'b': len(os.listdir(directory+"/b")),
              'c': len(os.listdir(directory+"/c")),
              'v': len(os.listdir(directory+"/v")),
             }
    
    # Printing the count in each set to the screen
    x_pos = 5

    cv2.putText(frame, "TRAIN/TEST : "+ test_or_train, (x_pos, 50), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.8, (0,0,255), 1)
    cv2.putText(frame, "IMAGE COUNT"+ str(add_up), (x_pos, 70), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.8, (0,0,255), 1)
    cv2.putText(frame, "A count: "+str(add_up['a']), (x_pos, 120), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.7, (255,0,0), 1)
    cv2.putText(frame, "B count: "+str(add_up['b']), (x_pos, 140), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.7, (255,0,0), 1)
    cv2.putText(frame, "C count: "+str(add_up['c']), (x_pos, 160), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.7, (255,0,0), 1)
    cv2.putText(frame, "V count: "+str(add_up['v']), (x_pos, 180), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.7, (255,0,0), 1)


 #------------------------------Source code for ROI--------------------------------------   
    # Coordinates of the ROI
    x1 = int(0.5*frame.shape[1])
    y1 = 10
    x2 = frame.shape[1]-10
    y2 = int(0.5*frame.shape[1])
    # Drawing the ROI
    # The increment/decrement by 1 is to compensate for the bounding box
    cv2.rectangle(frame, (x1-1, y1-1), (x2+1, y2+1), (255,0,0) ,1)
    # Extracting the ROI
    roi = frame[y1:y2, x1:x2]
    roi = cv2.resize(roi, (64,64)) #modified roi
 
    cv2.imshow("Frame", frame)
    
    roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    
    _, roi = cv2.threshold(roi, 120, 255, cv2.THRESH_BINARY)
    
    cv2.imshow("ROI", roi)
 #------------------------------source code end--------------------------------------   
   
  #------------------------------self modification--------------------------------------      
    keyboard_intrp = cv2.waitKey(10) # 10 milliseconds
    if keyboard_intrp & 0xFF == 27: # esc key
        break
    if keyboard_intrp & 0xFF == ord('a'):
        cv2.imwrite(directory+'a/'+str(add_up['a'])+'.jpg', roi)
    if keyboard_intrp & 0xFF == ord('b'):
        cv2.imwrite(directory+'b/'+str(add_up['b'])+'.jpg', roi)
    if keyboard_intrp & 0xFF == ord('c'):
        cv2.imwrite(directory+'c/'+str(add_up['c'])+'.jpg', roi)
    if keyboard_intrp & 0xFF == ord('v'):
        cv2.imwrite(directory+'v/'+str(add_up['v'])+'.jpg', roi)

real_time_capture.release()
cv2.destroyAllWindows()


# In[64]:


#directory = 'data/'+test_or_train+'/'
print(len(os.listdir('data/test'+"/v")))
print(len(os.listdir('data/train'+"/v")))


# In[ ]:




