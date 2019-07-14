#通过搭建卷积神经网络来实现sklearn库中的手写数字识别
#网络结构在md文件中
import tensorflow as tf
from sklearn.datasets import load_digits
import numpy as np

digits=load_digits()
X_data=digits.data.astype(np.float32)
#输入 1794x64
Y_data=digits.target.astype(np.float32).reshape(-1,1)
#输出（标签）(1794,1)=[0,1,2...9]

from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler()
X_data=scaler.fit_transform(X_data)
#归一化到0~1
#scale=(x-x.min)/(x.max-x.min)(百分比)
#_x=scale*(max-min)+min 
#默认min=0,max=1,即实现了调整到0~1

from sklearn.preprocessing import OneHotEncoder
Y=OneHotEncoder().fit_transform(Y_data).todense()
#OneHot编码，把离散特征变得更容易训练
#0,1,2->[1,0,0...],[0,1,0,...],[0,0,1...]
#归一化+和输出层(10neurons)一致
#todense是转化成一般的矩阵形式

X=X_data.reshape(-1,8,8,1)
#(batch,height,width,channels)
#把64长度的一维矩阵转化成8x8(图片)

batch_size=8#MBGD算法
def generate_batch(_X,_Y,_n,_batch_size):
    for _i in range(_n//_batch_size):#//整除
        start=_i*_batch_size
        end=start+_batch_size
        batch_xs=X[start:end]
        batch_ys=Y[start:end]
        yield batch_xs,batch_ys



