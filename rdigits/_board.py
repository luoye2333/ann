import tensorflow as tf

#定义命名空间
with tf.name_scope('input'):
    #fetch：就是同时运行多个op的意思
    input1 = tf.constant(3.0,name='A')#定义名称，会在tensorboard中代替显示
    input2 = tf.constant(4.0,name='B')
    input3 = tf.constant(5.0,name='C')
with tf.name_scope('op'):
    #加法
    add = tf.add(input2,input3)
    #乘法
    mul = tf.multiply(input1,add)
with tf.Session() as ss:
    #默认在当前py目录下的logs文件夹，没有会自己创建
    wirter = tf.summary.FileWriter('logs/',ss.graph)
    result = ss.run([mul,add])
    print(result)