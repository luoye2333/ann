import tensorflow as tf
import os
x=tf.Variable(tf.random_normal([1]))
with tf.Session() as sess:
    s=tf.train.Saver()
    sess.run(tf.global_variables_initializer())
    #print(os.path.dirname(__file__))
    s.save(sess,os.path.dirname(__file__)+'\\save\\1')