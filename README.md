# The-Simpsons-Characters-Recognition-Challenge
作法說明

流程圖
![image](https://github.com/107368009jungchengtsai/The-Simpsons-Characters-Recognition-Challenge/blob/master/%E6%B5%81%E7%A8%8B%E5%9C%96.jpg)
程式流程
    1.宣告和定義
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


    2.讀檔
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
