# The-Simpsons-Characters-Recognition-Challenge
作法說明

## 流程圖
![image](https://github.com/107368009jungchengtsai/The-Simpsons-Characters-Recognition-Challenge/blob/master/%E6%B5%81%E7%A8%8B%E5%9C%96.jpg)

## 1.宣告和定義

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
## 2.讀檔

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
## 3.split train_data/valid_data

    images, labels=read_main('train/characters-20')
    x_train, x_test, y_train, y_test = train_test_split(images, labels, test_size=0.1)
## 4.印出x_train,y_train和x_test,y_test資料數量

    print(x_train.shape)
    print(y_train.shape)
    print(x_test.shape)
    print(y_test.shape)
## 5.訓練出來的值乘上權重

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
## 6.查看model
    model.summary()
    model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
## 7.正規化
    
    datagen = ImageDataGenerator(zoom_range=0.2, width_shift_range=0.2, height_shift_range=0.2, horizontal_flip=True)
    datagen.fit(x_train)
## 8.設定跑的次數，和程式跑的過程
    
    epochs = 150
    batch_size = 256
    file_name = str(epochs) + '_' + str(batch_size)
    TB = TensorBoard(log_dir='logs/'+file_name)
    history=model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, shuffle=True, verbose=1, validation_data=(x_test, y_test),     callbacks=[TB])
    score = model.evaluate(x_test, y_test, verbose=0)
    print(score)
## 9.看val_acc,acc和val_loss,loss圖表
    
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
    ![image](https://github.com/107368009jungchengtsai/The-Simpsons-Characters-Recognition-Challenge/blob/master/%E5%9C%96%E8%A1%A8.png)
