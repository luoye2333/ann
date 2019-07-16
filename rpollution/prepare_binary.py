#生成一串01代码
#长度为20,要求其中1出现的次数
import csv
import random
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

with open("bdata.csv","w")as f:
    writer=csv.writer(f)
    writer.writerows(si)
with open("blabel.csv","w")as f:
    writer=csv.writer(f)
    writer.writerow(labels)
    #只有1维,不能writerows

