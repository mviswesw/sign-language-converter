#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# In[33]:


#import sys, os
import operator

import cv2
from keras.models import model_from_json

#get from model
json_file = open("saved_model.json", "r")
model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(model_json)

# load weights into new model
loaded_model.load_weights("saved_model.h5")
print("Model loaded from trained data")

real_time_capture = cv2.VideoCapture(0)


categories = {'A': 'a', 'B': 'b', 'C': 'c', 'V':'v'}



while True:
    _, frame = real_time_capture.read()
    # Simulating mirror image
    frame = cv2.flip(frame, 1)
    
    image = cv2.imread('amer_sign2.png')
    cv2.imshow("sign Language", image)

    
    # Got this from collect-data.py
    # Coordinates of the ROI
    x1 = int(0.5*frame.shape[1])
    y1 = 10
    x2 = frame.shape[1]-10
    y2 = int(0.5*frame.shape[1])
    # Drawing the ROI
    # The increment/decrement by 1 is to compensate for the bounding box
    cv2.rectangle(frame, (x1-1, y1-1), (x2+1, y2+1), (255,0,0) ,1)
    # Extracting the ROI
    roi = frame[y1+50:y2+50, x1:x2]
    
    # Resizing the ROI so it can be fed to the model for prediction
    roi = cv2.resize(roi, (64, 64)) 
    roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    _, test_image = cv2.threshold(roi, 120, 255, cv2.THRESH_BINARY)
    cv2.imshow("test", test_image)
    # Batch of 1
    result = loaded_model.predict(test_image.reshape(1, 64, 64, 1))
    
    #----------------------modifications------------------------------
    
    prediction = {'A': result[0][0], 
                  'B': result[0][1], 
                  'C': result[0][2],
                  'V': result[0][3]
                 }
    # Sorting based on top prediction
    prediction = sorted(prediction.items(), key=operator.itemgetter(1), reverse=True)
    
    # Displaying the predictions
    cv2.putText(frame, prediction[0][0], (10, 120), cv2.FONT_HERSHEY_COMPLEX_SMALL, 3, (255,0,0), 1)    
    cv2.imshow("Frame", frame)
    
    
    interrupt = cv2.waitKey(10)
    if interrupt & 0xFF == 27: # esc key
        break
        
real_time_capture.release()
cv2.destroyAllWindows()


# In[ ]:




