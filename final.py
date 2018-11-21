
# coding: utf-8

# In[2]:

import os,sys
import cv2
import numpy as np
from sklearn.cross_validation import train_test_split
from keras.models import Sequential, Model, load_model
from keras import applications
from keras.layers import *
from keras.callbacks import *
from keras.utils import np_utils
from keras.optimizers import SGD, Adam
from keras.preprocessing.image import ImageDataGenerator



images = []
labels = []
listdir = []

def read_images_labels(path,i):
    for file in os.listdir(path):
        abs_path = os.path.abspath(os.path.join(path, file))    
        if os.path.isdir(abs_path):
            i+=1                                               
            temp=os.path.split(abs_path)[-1]                   
            listdir.append(temp)                              
            read_images_labels(abs_path,i)                     
            amount=int(len(os.listdir(path))-1)                
            sys.stdout.write('\r'+'>'*(i*100//amount)+' '*((amount-i)*(100//amount) )+'[%s%%]'%(i*100/amount)+temp) 
        else:  
            if file.endswith('.jpg'):
                image=cv2.resize(cv2.imread(abs_path),(64,64)) 
                images.append(image)                           
                labels.append(i-1)                             
    return images, labels ,listdir

def read_main(path):
    images, labels ,listdir = read_images_labels(path,i=0)
    images = np.array(images,dtype=np.float32)/255
    labels = np_utils.to_categorical(labels, num_classes=20)
    np.savetxt('listdir.txt', listdir, delimiter = ' ',fmt="%s")
    return images, labels

images, labels=read_main('train/characters-20')
x_train, x_test, y_train, y_test = train_test_split(images, labels, test_size=0.1)

print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)


# In[3]:

model = Sequential()
model.add(Conv2D(64, kernel_size=3, padding='same', activation='relu', input_shape=x_train.shape[1:]))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.5))

model.add(Conv2D(128, kernel_size=3, padding='same', activation='relu'))
model.add(Conv2D(128, kernel_size=3, padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.5))

model.add(Conv2D(256, kernel_size=3, padding='same', activation='relu'))
model.add(Conv2D(256, kernel_size=3, padding='same', activation='relu'))
model.add(Conv2D(256, kernel_size=3, padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.5))

model.add(Flatten())
model.add(Dense(512,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(20,activation='softmax'))
model.summary()
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

datagen = ImageDataGenerator(zoom_range=0.2, width_shift_range=0.2, height_shift_range=0.2, horizontal_flip=True)
datagen.fit(x_train)

epochs = 150
batch_size = 256
file_name = str(epochs) + '_' + str(batch_size)
TB = TensorBoard(log_dir='logs/'+file_name)
history=model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, shuffle=True, verbose=1, validation_data=(x_test, y_test), callbacks=[TB])
#model.save('h5/' + file_name + '.h5')
score = model.evaluate(x_test, y_test, verbose=0)
print(score)


# In[4]:

import matplotlib.pyplot as plt
plt.plot(history.history['val_acc'])
plt.plot(history.history['acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

plt.plot(history.history['val_loss'])
plt.plot(history.history['loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


# In[5]:

model.save('model.h5')
del model
import numpy as np
import pandas
import pandas as pd
def read_images_labels(path,i):
    for file in os.listdir(path):
        abs_path = os.path.abspath(os.path.join(path, file))
        if os.path.isdir(abs_path):
            i+=1
            temp = os.path.split(abs_path)[-1]
            listdir.append(temp)
            read_images_labels(abs_path,i)
            amount = int(len(os.listdir(path)))
            sys.stdout.write('\r'+'>'*(i)+' '*(amount-i)+'[%s%%]'%(i*100/amount)+temp)
        else:
            if file.endswith('.jpg'):
                image = cv2.resize(cv2.imread(abs_path),(64,64))
                images.append(image)
                labels.append(i-1)

    return images, labels, listdir

def read_main(path):
    images, labels, listdir = read_images_labels(path,i=0)
    images = np.array(images,dtype=np.float32)/255
    labels = np_utils.to_categorical(labels, num_classes=20)
    return images, labels

def read_images(path):
    images=[]
    for i in range(990):
        image = cv2.resize(cv2.imread(path+str(i+1)+'.jpg'), (64,64))
        images.append(image)
    images = np.array(images,dtype=np.float32)/255
    return images

def transform(listdir,label,lenSIZE):
    label_str = []
    for i in range (lenSIZE):
        temp = listdir[label[i]]
        label_str.append(temp)
    return label_str

dataframe = pd.read_csv("listdir.txt",header=None)
dataset=dataframe.values
loadtxt=dataset[:,0]


images = read_images('test/test/')
print("\n",images.shape)
model = load_model('model.h5')
predict = model.predict_classes(images, verbose=1)
label_str = transform(loadtxt, predict ,images.shape[0])


raw_data={'id':range(1,991),
          'character':label_str    
}
df=pandas.DataFrame(raw_data,columns=['id','character'])
df.to_csv('test_score.csv',index=False,float_format='%.0f')


# In[ ]:



