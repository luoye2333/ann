# tensorboard --logdir=.\logs
#

import tensorflow as tf
import numpy as np

with tf.name_scope('data'):
    x_data = np.random.rand(100).astype(np.float32)
    y_data = 0.3*x_data+0.1
with tf.name_scope('parameters'):
    with tf.name_scope("weights"):
        weight = tf.Variable(tf.random_uniform([1],-1.0,1.0))
        tf.summary.histogram("weight",weight)
    with tf.name_scope("biases"):
        bias = tf.Variable(tf.zeros([1]))
        tf.summary.histogram('bias',bias)
with tf.name_scope('y_prediction'):
    y_prediction = weight*x_data+bias
with tf.name_scope('loss'):
    loss = tf.reduce_mean(tf.square(y_data-y_prediction))
    tf.summary.scalar('loss',loss)
##creat optimizer
optimizer = tf.train.GradientDescentOptimizer(0.5)
#creat train ,minimize the loss 
with tf.name_scope('train'):
    train = optimizer.minimize(loss)


sess = tf.Session()
merged=tf.summary.merge_all()
writer = tf.summary.FileWriter("logs/", sess.graph)
sess.run(tf.global_variables_initializer())
## Loop
for step  in  range(101):
    sess.run(train)
    rs=sess.run(merged)
    writer.add_summary(rs,step)
    if step %10==0 :
        print(step,'weight:',sess.run(weight),'bias:',sess.run(bias))