#生成一串01代码
#长度为20,要求其中1出现的次数
import pandas as pd
import random
import os

path=os.path.dirname(__file__)

len=20
#含1个数从0~len,每组n个,共(len+1)*n
sample_num=123
si=[]
labels=[]

n1=-1
for _ in range(len+1):
    n1=n1+1
    for _ in range(sample_num//(len+1)):#//整除
        s=[]
        for i in range(len):
            if i<n1:
                s.append(1)
            else:
                s.append(0)
        random.shuffle(s)
        si.append(s)
        labels.append(n1)
#print(si,labels)
ds=pd.DataFrame(si)
dl=pd.DataFrame(labels)
# writer_d=pd.ExcelWriter(path+"/bdata.xlsx")
# writer_l=pd.ExcelWriter(path+"/blabel.xlsx")
# ds.to_excel(writer_d)
# dl.to_excel(writer_l)
# writer_d.save()
# writer_d.close()
# writer_l.save()
# writer_l.close()
#使用pandas的dataframe把数据保存在xlsx中

ds.to_csv(path+"/bdata.csv")
dl.to_csv(path+"/blabel.csv")
#使用pandas的dataframe把数据保存在csv中

#csv可以直接预览,并且更简单……