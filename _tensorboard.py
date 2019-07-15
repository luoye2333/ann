import tensorflow as tf
import os
x=tf.Variable(tf.random_normal([2]))
y=tf.reduce_mean(x)
tf.summary.scalar('y',y)
merged=tf.summary.merge_all()

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    writer=tf.summary.FileWriter(
        os.path.dirname(__file__)+"\\log",sess.graph)
    summary,y_=sess.run([merged,y])
    writer.add_summary(1,y_)



