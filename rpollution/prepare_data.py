from pandas import read_csv
from datetime import datetime

#load data
def parse(x):
    return datetime.strptime(x,'%Y %m %d %H')

dataset=read_csv('PM2d5.csv',parse_dates=[['year','month','day','hour']],index_col=0,date_parser=parse)
dataset.drop('No',axis=1,inplace=True)
#manually specify column names
dataset.columns=['pollution','dew','Temp','Pres','wind_dir','wind_spd','snow','rain']
dataset.index.name='date'
#mark all NA values with 0
dataset['pollution'].fillna(0,inplace=True)
d=dataset[24:]
print(d.head(5))
dataset.to_csv('pollution.csv')