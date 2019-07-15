#通过搭建卷积神经网络来实现sklearn库中的手写数字识别
#网络结构在md文件中

#训练结果 
#epoch train test
#1100  0.998 0.906
import datetime
import os

import numpy as np
import tensorflow as tf
from sklearn.datasets import load_digits
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

digits=load_digits()
X_data=digits.data.astype(np.float32)
#输入 1794x64
Y_data=digits.target.astype(np.float32).reshape(-1,1)
#输出（标签）(1794,1)=[0,1,2...9]

scaler=MinMaxScaler()
X_data=scaler.fit_transform(X_data)
#归一化到0~1
#scale=(x-x.min)/(x.max-x.min)(百分比)
#_x=scale*(max-min)+min 
#默认min=0,max=1,即实现了调整到0~1

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
        #end不需要-1,因为x[0:len]返回0~len-1
        batch_xs=X[start:end]
        batch_ys=Y[start:end]
        yield batch_xs,batch_ys

tf.reset_default_graph()
#input
tf_X=tf.placeholder(tf.float32,[None,8,8,1])
tf_Y=tf.placeholder(tf.float32,[None,10])#长度不固定

conv_filter_w1=tf.Variable(tf.random_normal([3,3,1,10]))#卷积核
conv_filter_b1=tf.Variable(tf.random_normal([10]))#biase共用
relu_feature_maps1=tf.nn.relu(
    tf.nn.conv2d(tf_X,conv_filter_w1,
    strides=[1,1,1,1],padding="SAME")
    +conv_filter_b1
)

max_pool1=tf.nn.max_pool(
    relu_feature_maps1,
    ksize=[1,3,3,1],
    strides=[1,2,2,1],
    padding="SAME")
#print max_pool1
#Tensor("MaxPool:0", shape=(?, 4, 4, 10), dtype=float32)

conv_filter_w2=tf.Variable(tf.random_normal([3,3,10,5]))
conv_filter_b2=tf.Variable(tf.random_normal([5]))
conv_out2=tf.nn.conv2d(
    relu_feature_maps1,conv_filter_w2,
    strides=[1,2,2,1],padding="SAME")+conv_filter_b2

#Tensor("add_4:0", shape=(?, 4, 4, 5), dtype=float32)

batch_mean,batch_var=tf.nn.moments(conv_out2,[0,1,2],keep_dims=True)
shift=tf.Variable(tf.zeros([5]))
scale=tf.Variable(tf.ones([5]))
epsilon=1e-3
BN_out=tf.nn.batch_normalization(
    conv_out2,batch_mean,batch_var,shift,scale,epsilon
)
#print BN_out
relu_BN_maps2=tf.nn.relu(BN_out)
#Tensor("batchnorm/add_1:0", shape=(?, 4, 4, 5), dtype=float32)
max_pool2=tf.nn.max_pool(
    relu_BN_maps2,ksize=[1,3,3,1],
    strides=[1,2,2,1],
    padding="SAME"
)
#print max_pool2
#Tensor("MaxPool_1:0", shape=(?, 2, 2, 5), dtype=float32)
max_pool2_flat=tf.reshape(max_pool2,[-1,2*2*5])

fc_w1=tf.Variable(tf.random_normal([2*2*5,50]))
fc_b1=tf.Variable(tf.random_normal([50]))
fc_out1=tf.nn.relu(tf.matmul(max_pool2_flat,fc_w1)+fc_b1)

out_w1=tf.Variable(tf.random_normal([50,10]))
out_b1=tf.Variable(tf.random_normal([10]))
prediction=tf.nn.softmax(tf.matmul(fc_out1,out_w1)+out_b1)

#########################################
loss=-tf.reduce_mean(
    tf_Y*tf.log(tf.clip_by_value(prediction,1e-11,1.0))
)
train_step=tf.train.AdamOptimizer(1e-3).minimize(loss)
y_pred=tf.arg_max(prediction,1)
bool_pred=tf.equal(tf.arg_max(tf_Y,1),y_pred)
accuracy=tf.reduce_mean(tf.cast(bool_pred,tf.float32))

tf.summary.scalar('accuracy',accuracy)
merged=tf.summary.merge_all()

##############################################
epoch_to_train=3

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    train_writer=tf.summary.FileWriter(
        os.path.dirname(__file__)+"\\summary\\train",sess.graph)
    test_writer=tf.summary.FileWriter(
        os.path.dirname(__file__)+"\\summary\\test")

    s=tf.train.Saver()
    if os.path.exists(os.path.dirname(__file__)+"\\save4\\"):
        s.restore(sess,tf.train.latest_checkpoint(
            os.path.dirname(__file__)+"\\save4\\"))
    #s.restore(sess,os.path.dirname(__file__)+"\\save2\\+200@2019-07-14-19-26-17")
    for epoch in range(epoch_to_train):
        #for batch_xs,batch_ys in generate_batch(X,Y,Y.shape[0],batch_size):
        #1700做训练,97个做测试
        for batch_xs,batch_ys in generate_batch(X,Y,1700,batch_size):
            sess.run(train_step,feed_dict={
                tf_X:batch_xs,tf_Y:batch_ys
            })
        #if (epoch%10==0):
        if True:
            summary,acc_train=sess.run([merged,accuracy],feed_dict={tf_X:X[0:1700],tf_Y:Y[0:1700]})
            test_writer.add_summary(epoch,acc_train)
            #res_ypred=y_pred.eval(feed_dict={tf_X:X[1700:1796],tf_Y:Y[1700:1796]}).flatten()
            #print(res_ypred)
            #acc_validate=accuracy_score(Y_data[1700:1796],res_ypred.reshape(-1,1))
            summary,acc_validate=sess.run([merged,accuracy],feed_dict={tf_X:X[1700:1796],tf_Y:Y[1700:1796]})
            train_writer.add_summary(epoch,acc_validate)
            print("{0}  {1:.3f}  {2:.3f}".format(epoch,acc_train,acc_validate))
    
    
    
    s.save(sess,os.path.dirname(
        __file__)+"\\save4\\"+"+"+str(epoch_to_train)+"@"
        +datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    )
