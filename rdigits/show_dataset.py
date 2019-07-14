from sklearn.datasets import load_digits
digits = load_digits()
print(digits.data.shape)
#(1797,64)
#共1797个样本，每个为1x64
#digits.images[0]为(8,8)
import matplotlib.pyplot as plt
plt.gray()#表明这是一张灰度图
#数值的大小表明颜色深浅，而不是颜色
plt.matshow(digits.images[0]) 
#把矩阵加载到图中，内存中已有这张图的实例
plt.show()