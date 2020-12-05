# Electric-grid-stability-som
 Design and implement the SOM model on the given data set. SOM(Self Organising Maps) which is used for unsupervised learning. And perform the visualisation of the respective trained network .
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

data=pd.read_csv('egss.csv')
data

#Creating File
sha=data.shape
print(sha)
data.isnull().sum()
data.describe()

si=data.groupby('stabf').size()
si

plt.figure(figsize=(15,10))
data1=pd.DataFrame(data[['tau1','p1','g1','stab','stabf']])
data1.plot(kind='density',subplots=True,sharex=False)
plt.show()

#data taking
data=pd.read_csv('egss.csv')
print(data.head())
y=data['stabf'].values
print("Target: ",y)
X=data.drop(['stabf'],axis=1)
y=np.where(y=='stable',1,-1)

print("\nInput:\n",X)
print("Target: ",y)

#scaling data to some set for better training the model    
from sklearn.preprocessing import StandardScaler
scaler=StandardScaler().fit(X)
rescaledX=scaler.transform(X)
rescaledX[0:5,:]

#Splitting data
#we have methods to split data some of them are 
#1. TRain test split
#2. sampling : we can split data using sampling method from numpy libray rather than sklearn

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(
        rescaledX,y,test_size=0.3,random_state=0)
        
        
  ##developing model
w =np.array([[2 ,8], [0.6, 4], [0.5, 7], [0.9, 0.3],[0.4,0.8],[0.43,0.2],[1,0.6],[0.9,.3],[.6,7],[.3,.8],[6,.7],[.7,.8],[.9,.8]])
lrate= 0.6
e=1
D=[0,0]
distance=[]
print('learning rate of this epoch is',lrate);
while(e<=3): # e is epoch
    print('Epoch is',e);
    
    for i in range(13): # number of patterns 13
        for j in range(2): # size of neurons 
             temp=0
             for k in range(13):
                temp = temp + ((w[k,j]-X_train[i,k])**2)
             D[j]=temp # distance matrix
             distance.append(temp)
        #decide winner neurons
        if(D[0]<D[1]):
            J=0
        else:
            J=1
        
        print('winning unit is',J+1)
        print('weight updation ...')
        for m in range(4):
             w[m,J]=w[m,J] + (lrate *(X_train[i,m]-w[m,J]))
        print('Updated weights',w)
        

    e=e+1
    lrate = 0.5*lrate;
    print(' updated learning rate after ',e,' epoch is',lrate)
    
    t=np.arange(100)
plt.scatter(distance, distance, marker='o');
