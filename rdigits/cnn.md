搭建的卷积神经网络结构
1.input:28x28x1(深度为1，黑白图像)
2.conv1: 
    filter 3x3x1x10(10个不同的卷积核)
    stride 1x1(上下步长)
    output 28+1-3=26
           26x26x10
3.Relu1
4.max pooling1
    ksize:3x3
    stride:2x2
    output:(26-3+2)/2=12+1
            13x13x10
5.conv2
    filter:3x3x10(深度10)x5
    stride:2x2
    output:(13-3+2)/2=6
            6x6x5
6.BN(batch normalization)
    减去一批样本中的均值，然后除以方差
    归一化
    output:6x6x5
7.Relu2
8.max pooling2
    ksize:3x3
    stide:2x2
    output:(6-3+2)/2=2+1
            3x3x5
9.FC(full connection)
    neurons:50
    weight:45x50
10.FC2
    neurons:10
    weight:50x10
