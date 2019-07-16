
import numpy as np
from random import shuffle
 
train_input = ['{0:020b}'.format(i) for i in range(2**20)]
#生成01的全排列,2^20个
#可以不生成全排列,保证1的数量平均分布即可
shuffle(train_input)

#convert to list
#耗时较多
train_input=[map(int,i) for i in train_input]
ti=[]
for i in train_input:
    temp_list=[]
    for j in i:
        temp_list.append([j])
    ti.append(np.array(temp_list))
train_input=ti

train_output = []

#onehot
#耗时很多,而且根本不需要onehot编码
for i in train_input:
    count = 0
    for j in i:
        if j[0] == 1:
            count+=1
    temp_list = ([0]*21)
    temp_list[count]=1
    train_output.append(temp_list)

NUM_test=int(0.009*(2**20))#0.9%用来测试
test_input = train_input[NUM_test:]
test_output = train_output[NUM_test:]
 
train_input = train_input[:NUM_test]
train_output = train_output[:NUM_test]
#############################################
import tensorflow as tf
data=tf.placeholder(tf.float32,[None,20,1])
target=tf.placeholder(tf.float32,[None,21])

num_hidden_unit=24
cell=tf.nn.rnn_cell.LSTMCell(
    num_hidden_unit,state_is_tuple=True)
val,state=tf.nn.dynamic_rnn(cell,data,dtype=tf.float32)

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
output=tf.matmul(last,weight)+bias
prediction=tf.nn.softmax(output)
cross_entropy=-tf.reduce_sum(
    target*tf.log(tf.clip_by_value(prediction,1e-10,1.0)))
optimizer=tf.train.AdamOptimizer()
minimize=optimizer.minimize(cross_entropy)

mistakes=tf.not_equal(
    tf.argmax(target,1),tf.argmax(prediction,1))
error=tf.reduce_mean(tf.cast(mistakes,tf.float32))

###################################3
init_opr=tf.initialize_all_variables()
sess=tf.Session()
sess.run(init_opr)

batch_size=1000
#批次数量是否太大?
num_of_batches=int(len(train_input)/batch_size)
epoch_to_train=1000
for epoch in range(epoch_to_train):
    ptr=0
    for j in range(num_of_batches):
        inp=train_input[ptr:ptr+batch_size]
        out=train_output[ptr:ptr+batch_size]
        ptr+=batch_size
        sess.run(minimize,feed_dict={
            data:inp,target:out
        })
    print("epoch ",epoch)
    #在此添加每次训练完后的train test accuracy
incorrect=sess.run(error,feed_dict={
    data:test_input,target:test_output
})
print('Epoch {:2d} error {:3.1f}%'.format(epoch + 1, 100 * incorrect))
sess.close()

#需要添加tensorboard支持
#lstm层数还要增加

