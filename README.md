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
## 2.train_data讀檔
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
 ![image](https://github.com/107368009jungchengtsai/The-Simpsons-Characters-Recognition-Challenge/blob/master/model%E6%95%B8%E9%87%8F.png)
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
  ![image](https://github.com/107368009jungchengtsai/The-Simpsons-Characters-Recognition-Challenge/blob/master/process.png)
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
## 10.model存檔
    model.save('model.h5')
## 11.test_data讀檔
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
## 12.test丟進model預測出price,結果存進csv檔案
    images = read_images('test/test/')
    print("\n",images.shape)
    model = load_model('model.h5')
    predict = model.predict_classes(images, verbose=1)
    label_str = transform(loadtxt, predict ,images.shape[0])
    raw_data={'id':range(1,991),'character':label_str}
    df=pandas.DataFrame(raw_data,columns=['id','character'])
    df.to_csv('test_score.csv',index=False,float_format='%.0f')
 ![image](https://github.com/107368009jungchengtsai/The-Simpsons-Characters-Recognition-Challenge/blob/master/images.shape.png)
## 13.Kaggle排名
## 14.分析
    這一次參考各種殼層寫法，發覺現在使用的這種表現較佳，做資料分析時，有把表現不佳的資料剃除，可是表現不佳，所以最後選擇不做剃除，可能是資料量太少的     關系，epochs提高有助於準確度。
## 15.改進
    這一次的表現都算不錯，不過還沒試過pre-train的model，似乎是用pre-train的model正確率可以達到100%，下一次的作業可以使用看看pre-train的model，不     過老師上課也說過，環境不同，用自己的比較符合自己的需求。
    
