import pandas as pd
import numpy as np
import tensorflow as tf
import os
from sklearn.preprocessing import MinMaxScaler,OneHotEncoder
import random

path=os.path.dirname(__file__)

data=pd.read_csv(path+"\\bdata.csv",index_col=0)
#index_col=0去除第一列的行号
X_data=data.values
#去除第一行列名称,并转化成nparray
data=pd.read_csv(path+"\\blabel.csv",index_col=0)
Y_data=data.values
#####################################
sample_num=data.shape[0]
sample_train=sample_num*4//5
sample_test=sample_num-sample_train

#标签不需要onehotencoder,因为是累积过程
#把标签y归一化到0~1
scaler=MinMaxScaler()
Y_data=scaler.fit_transform(Y_data)


#生成batch
#注意用到了随机抽取
batch_size=8
def generate_batch(_X,_Y,_n,_batch_size):
    batch_xs=[]
    batch_ys=[]
    for _ in range(_batch_size):
        r=random.randint(0,_n-1)
        batch_xs=np.append(batch_xs,_X[r],axis=0)
        batch_ys=np.append(batch_ys,_Y[r],axis=0)

    yield batch_xs,batch_ys

time_step=20
rnn_unit=8#每层lstm的单元数
lstm_layer=2
LearningRate=0.006
input_size=1
output_size=1

tf.reset_default_graph()
X=tf.placeholder(tf.float32,[None,time_step,input_size])
Y=tf.placeholder(tf.float32,[None,time_step,output_size])
with tf.name_scope("weight"):
    w1=tf.Variable(tf.random_normal([input_size,rnn_unit]))
    w2=tf.Variable(tf.random_normal([rnn_unit,output_size]))
with tf.name_scope("biases"):
    b1=tf.Variable(tf.random_normal([rnn_unit,]))
    b2=tf.Variable(tf.random_normal([output_size,]))

def lstm(batch):
    _input=tf.reshape(X,[-1,input_size])
    input_rnn=tf.matmul(_input,w1)+b1
    input_rnn=tf.reshape(input_rnn,[-1,time_step,rnn_unit])
    cell=tf.nn.rnn_cell.MultiRNNCell(
        tf.nn.rnn_cell.BasicLSTMCell(rnn_unit)
    )
    init_state=cell.zero_state(batch,dtype=tf.float32)
    output_rnn,final_states=tf.nn.dynamic_rnn(
        cell,input_rnn,initial_state=init_state,dtype=tf.float32
    )
    output=tf.reshape(output_rnn,[-1,rnn_unit])
    pred=tf.matmul(output,w2)+b2
    return pred,final_states

def train_lstm():
    global batch_size
    
    epoch_to_train=100

    with tf.variable_scope("sec_lstm"):
        pred,_=lstm(batch_size)
    loss=tf.reduce_mean(tf.square(
        tf.reshape(pred,[-1])-tf.reshape(Y,[-1])
    ))
    train_opr=tf.train.AdamOptimizer(LearningRate).minimize(loss)
    saver=tf.train.Saver()
    loss_list=[]
    with tf.Session as sess:
        sess.run(tf.global_variables_initializer())
        for epoch in range(epoch_to_train):
            for xb,yb in generate_batch(X_data,Y_data,sample_test,batch_size):
                _,_loss=sess.run([train_opr,loss],feed_dict={
                    X:xb,Y:yb
                })
            


