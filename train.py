#!/usr/bin/env python
# coding: utf-8

# In[62]:


#-------------------source code begin-----------------------------------------
# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

model_classifier = Sequential()
model_classifier.add(Convolution2D(32, (3, 3), input_shape=(64, 64, 1), activation='relu'))
model_classifier.add(MaxPooling2D(pool_size=(2, 2)))
model_classifier.add(Convolution2D(32, (3, 3), activation='relu'))

model_classifier.add(MaxPooling2D(pool_size=(2, 2)))
model_classifier.add(Flatten())


model_classifier.add(Dense(units=128, activation='relu'))
model_classifier.add(Dense(units=4, activation='softmax')) # softmax for more than 2

# Compile CNN
model_classifier.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy']) # categorical_crossentropy for more than 2


# In[63]:


from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory('data/train',
                                                 target_size=(64, 64),
                                                 batch_size=5,
                                                 color_mode='grayscale',
                                                 class_mode='categorical')

test_set = test_datagen.flow_from_directory('data/test',
                                            target_size=(64, 64),
                                            batch_size=5,
                                            color_mode='grayscale',
                                            class_mode='categorical') 
#-------------------source code end-----------------------------------------


# In[64]:


#-------------modified source code-----------------
model_classifier.fit_generator(
        training_set,
        steps_per_epoch=200, # Number of images in training set
        epochs=10,
        validation_data=test_set,
        validation_steps=16)# Number of images in test set


# Saving the model
model_json = model_classifier.to_json()
with open("saved_model.json", "w") as json_file:
    json_file.write(model_json)
    
model_classifier.save_weights('saved_model.h5')


# In[ ]:





# In[ ]:




