#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  3 18:29:44 2018

@author: sand_boa
"""
import numpy as np
import xml.etree.ElementTree as ET
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

tree = ET.parse('renjunet_v10_20180803.xml')
root = tree.getroot()
print(root)


classifier = Sequential()

# Step 1 - Convolution
classifier.add(Conv2D(32, (5, 5), padding = 'same', input_shape = (15, 15, 3), activation = 'relu'))

# Adding a second convolutional layer
classifier.add(Conv2D(64, (3, 3),padding = 'same', activation = 'relu'))
#classifier.add(Conv2D(15, (3, 3), activation = 'relu', data_format="channels_last"))
#classifier.add(Conv2D(15, (3, 3), activation = 'relu', data_format="channels_last"))
#classifier.add(Conv2D(15, (3, 3), activation = 'relu', data_format="channels_last"))
#classifier.add(Conv2D(15, (3, 3), activation = 'relu', data_format="channels_last"))
#classifier.add(Conv2D(15, (3, 3), activation = 'relu', data_format="channels_last"))
#classifier.add(Conv2D(15, (3, 3), activation = 'relu', data_format="channels_last"))
#classifier.add(Conv2D(15, (3, 3), activation = 'relu', data_format="channels_last"))
#classifier.add(Conv2D(15, (3, 3), activation = 'relu', data_format="channels_last"))
classifier.add(Conv2D(128, (3, 3), padding='same', activation = 'relu'))
#classifier.add(Conv2D(255, (3, 3), activation = 'relu', data_format="channels_last"))
classifier.add(Conv2D(3, (1, 1), padding = 'same', activation = 'softmax'))

#classifier.add(Flatten())

# Step 4 - Full connection
#classifier.add(Dense(units = 128, activation = 'relu'))
#classifier.add(Dense(units = 1, activation = 'softmax'))

# Compiling the CNN
classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])



result = []
for data in root.iter('move'):
    val=[]
    board  = np.zeros((15,15))
    #print(board)
    data = data.text.split(" ")
    count = 0
    for stri in data:
        count+=1
        state= stri[:1]
        action =stri[1:]
        #print(type(action),type(state))
        try:
            if count%2:
                board[int(ord(state))-97][int(action)-1] = 1
            else:
                board[int(ord(state))-97][int(action)-1] = 2
        except (ValueError, TypeError) as e:
            print(e)
        encoded = to_categorical(board, num_classes=3)
        
        encoded.tolist()
        result.append(encoded)
        
result = np.array(result)
        
        #val.append((state,action))
    #result.append(val)
#import csv

#with open("output.csv", "w") as f:
#    writer = csv.writer(f)
#    writer.writerows(result)

#with open('output.data', 'w') as f:
   # pickle.dump(data, f)
   
   
print(result[:1500000].shape)

classifier.fit(result[:(((result.shape)[0])//2)],result[(((result.shape)[0])//2):], batch_size = 50, epochs = 1)
model_json = classifier.to_json()
with open('model.json', 'w') as json_file:
    json_file.write(model_json)
classifier.save_weights("Gomoku.h5")
import numpy as np
boardcheck  = np.zeros((15,15))
#print(boardcheck)
boardcheck[2][3]= 1
boardcheck[2][4]= 2
boardcheck[3][3]= 1
boardcheck[2][5]= 2
boardcheck[2][6]= 2
boardcheck[3][4]= 1
boardcheck[2][7]= 2
print(boardcheck)
boardcheck = to_categorical(boardcheck, num_classes=3)
ast=np.expand_dims(boardcheck, axis = 0)
pred = np.squeeze(classifier.predict(ast))
print(pred)
print(np.sum(pred, axis= 2))