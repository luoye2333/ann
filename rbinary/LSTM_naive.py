import numpy as np
import random
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import os
import datetime
print("import successfully")
print()
print()

path=os.path.dirname(__file__)
data=pd.read_csv(path+"\\bdata.csv",index_col=0)
labels=pd.read_csv(path+"\\blabel.csv",index_col=0)
data=data.values
labels=labels.values
sample_num=data.shape[0]
#shuffle
indices=np.arange(sample_num)
np.random.shuffle(indices)
data=data[indices]
labels=labels[indices]

#把二维矩阵化成三维,适用训练的函数
ti=[]
for i in range(sample_num):
    temp=[]
    for j in range(20):
        temp.append([data[i][j]])
    ti.append(np.array(temp))
train_input=ti
train_output=labels
#把label归一化0~20 -> 0~1
scaler=MinMaxScaler()
train_output=scaler.fit_transform(train_output)


num_test=int(0.01*sample_num)

test_input = train_input[:num_test]
test_output = train_output[:num_test]
train_input = train_input[num_test:]
train_output = train_output[num_test:]

 

print()
print()
print("preprocess successfully")
print()
print()
#############################################

import tensorflow as tf
data=tf.placeholder(tf.float32,[None,20,1])
target=tf.placeholder(tf.float32,[None,1])

num_hidden_unit=24
cell=tf.nn.rnn_cell.LSTMCell(
    num_hidden_unit,state_is_tuple=True)
val,_=tf.nn.dynamic_rnn(cell,data,dtype=tf.float32)

val=tf.transpose(val,[1,0,2])
last=tf.gather(val, int(val.get_shape()[0]) - 1)

weight=tf.Variable(tf.truncated_normal(
    [num_hidden_unit , int(target.get_shape()[1])]
    ))
bias=tf.Variable(
    tf.constant(0.1,shape=[
        target.get_shape()[1]
    ])
)
prediction=tf.matmul(last,weight)+bias
cross_entropy=tf.reduce_sum(
    tf.abs(target-tf.clip_by_value(prediction,1e-10,1.0)))
optimizer=tf.train.AdamOptimizer()
minimize=optimizer.minimize(cross_entropy)

pred=tf.round(prediction*20)/20.0
mistakes=tf.not_equal(
    target,pred
)
error=tf.reduce_mean(tf.cast(mistakes,tf.float32))
#summary
tf.summary.scalar('error',error)
init_opr=tf.initialize_all_variables()

print()
print()
print("start training")
print()
print()

sess=tf.Session()
sess.run(init_opr)
merged=tf.summary.merge_all()
train_writer=tf.summary.FileWriter(
    os.path.dirname(__file__)+"\\summary\\train",sess.graph)
test_writer=tf.summary.FileWriter(
    os.path.dirname(__file__)+"\\summary\\test")

##############        
s=tf.train.Saver()
save_num=1
save_dir=os.path.dirname(__file__)+"\\save"+str(save_num)+"\\"
if os.path.exists(save_dir):
    s.restore(sess,tf.train.latest_checkpoint(save_dir))


batch_size=10
#num_of_batches=int(len(train_input)/batch_size)
num_of_batches=10
epoch_to_train=100
for epoch in range(epoch_to_train):
    ptr=0
    for j in range(num_of_batches):
        inp=train_input[ptr:ptr+batch_size]
        out=train_output[ptr:ptr+batch_size]
        ptr+=batch_size
        sess.run(minimize,feed_dict={
            data:inp,target:out
        })
    #if epoch%10==0:
    if True:
        summary,error_train=sess.run([merged,error],
            feed_dict={data:train_input,
            target:train_output})
        train_writer.add_summary(summary,epoch)
        
        summary,error_test=sess.run([merged,error],
            feed_dict={data:test_input,
            target:test_output})
        test_writer.add_summary(summary,epoch)
        
        print("{0} {1:.4f} {2:.4f}".format(epoch,error_train,error_test))

s.save(sess,save_dir+"+"+str(epoch_to_train)+"@"
        +datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    )


incorrect=sess.run(error,feed_dict={
    data:test_input,target:test_output
})

incorrect = sess.run(error,{data: test_input, target: test_output})
print( sess.run(prediction,{data: [[[1],[0],[0],[1],[1],[0],[1],[1],[1],[0],[1],[0],[0],[1],[1],[0],[1],[1],[1],[0]]]}) )
print('Epoch {:2d} error {:3.1f}%'.format(i + 1, 100 * incorrect))

sess.close()

#lstm层数还要增加
#tensorboard --logdir=train:"./summary/train",test:"./summary/test"




