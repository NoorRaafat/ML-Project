#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
import seaborn as sns
data=pd.read_csv("data.csv")
data


# In[2]:


data.shape


# In[3]:


data.info()


# In[4]:


data.columns


# In[5]:


data.isnull().sum()# m= malignant خبيث b=benignحميد


# In[6]:


#data.drop('id',axis=1,inplace=True)
data.drop('Unnamed: 32',axis=1,inplace=True)


# In[7]:


data.isnull().sum()


# In[8]:


data


# In[9]:


data.diagnosis.unique()


# In[10]:


data["diagnosis"].value_counts()


# In[11]:


data['diagnosis'] = data['diagnosis'].map({'M':1,'B':0})#manuplution sample #rating
data.head()#عشان ارجع ب اول خمس قيم ف العمود 


# In[12]:


data.describe() #describe columns of numbers only


# In[13]:


data.groupby(by="diagnosis").count()


# In[14]:


l=data.groupby(by="diagnosis")
B=l.get_group(0)
B


# In[15]:


h=data.groupby(by="diagnosis")
M=l.get_group(1)
M


# In[16]:


data.groupby(by=["diagnosis","radius_mean"]).count()


# In[17]:


data[["diagnosis","radius_mean","texture_mean"]].sort_values(by="radius_mean")


# In[18]:


data[["diagnosis","radius_mean","texture_mean"]].sort_values(by="texture_mean")#عشان أثبت ان حجم النسيج بيعتمد علي نصف قطره اما بيبقي قليل التاني بيبقي قليل


# In[19]:


data.loc[data.radius_mean<10]#عشان أثبت ان اللي اقل من عشره ف نصف قطر الخليه دايما خبيث


# In[20]:


data.loc[data.radius_mean>10]#عشان اثبت ان اللي اكبر من عشره نصف قطره بيبقي حميد


# In[21]:


my_labels=data.diagnosis.unique().tolist()
my_labels


# In[22]:


data.diagnosis.value_counts()


# In[23]:


values=data.diagnosis.value_counts().tolist()
values


# In[24]:


data.hist()
plt.show()


# In[25]:


data.radius_mean.hist()


# In[26]:


plt.bar(data.radius_mean,data.texture_mean)
plt.show()


# In[27]:


data.plot(kind="bar")
plt.show()


# In[28]:


my_explode=[0,.1]# pipe chart
my_labels=['B','M']
my_colors=["green","red"]
plt.pie(values,labels=my_labels,startangle=90,explode=my_explode,colors=my_colors)
plt.show()


# In[29]:


x=data.loc[:301,["radius_mean"]]#independent #input   # linear    regression  supervised continous
y=data.loc[:301,["perimeter_mean"]]# depndent dataمعتمده علي اللي فوقها   #y=a+bx #output
x_test=data.loc[:302,["radius_mean"]]
y_desired=data.loc[:302,["perimeter_mean"]]#actual predicted data
x


# In[30]:


plt.scatter(x,y)
plt.show()


# In[31]:


from sklearn.linear_model import LinearRegression#calling model


# In[32]:


model=LinearRegression()#object from linear


# In[33]:


model.fit(x,y)#training


# In[34]:


y_predicted=model.predict(x_test)#test for predictiong
y_predicted #الداتا اللي المودل توقعها ليا 


# In[35]:


plt.scatter(x_test,y_predicted)
plt.show()


# In[36]:


plt.scatter(x_test,y_desired)
plt.show()


# In[37]:


model.intercept_ #التقاطع


# In[38]:


model.coef_ 


# In[39]:


import sklearn.metrics as mc


# In[40]:


dir(mc)  


# In[41]:


mc.mean_absolute_error(y_desired,y_predicted)#الفرق بين الاتنين ويجمعهم ويحطهم جوا المقياس ويقسم التموسط علي عددهم


# In[42]:


mc.mean_squared_error(y_desired,y_predicted)# الفرق بينهم ويجمعهم ويربعهم علي عددهم


# In[43]:


#polynomial


# In[44]:


from sklearn.preprocessing import PolynomialFeatures 


# In[45]:


polyF=PolynomialFeatures(degree=3)


# In[46]:


x_poly=polyF.fit_transform(x) 


# In[47]:


model=LinearRegression()


# In[48]:


model.fit(x_poly,y)


# In[49]:


y_poly_predict=model.predict(x_poly)


# In[50]:


plt.scatter(x,y)
plt.plot(x,y_poly_predict)


# In[51]:


from sklearn.pipeline import Pipeline #2 الطريقه المختصره للبولي 


# In[52]:


input=[('polynomial',PolynomialFeatures(degree=3)),('model',LinearRegression())]


# In[53]:


pip=Pipeline (input)


# In[54]:


pip.fit(x,y)


# In[55]:


y_poly_predict=pip.predict(x)
y_poly_predict


# In[56]:


plt.scatter(x,y)
plt.plot(x,y_poly_predict)


# In[57]:


# mulite linerregression


# In[58]:


x=data[["radius_mean","perimeter_mean"]]#y=a+b1x1+b2x2 ....  # input
y=data.area_mean # اللي هتوقعها #output


# In[59]:


from sklearn.model_selection import train_test_split


# In[60]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=.3)
x_train


# In[61]:


sns.pairplot(data,x_vars=["radius_mean","perimeter_mean"],y_vars=["area_mean"],diag_kind=None)
plt.show()


# In[62]:


from sklearn.linear_model import LinearRegression


# In[63]:


model=LinearRegression()#object from linear


# In[64]:


model.fit(x,y)#training


# In[65]:


y_predicted=model.predict(x_test)#test for predictiong
y_predicted #الداتا اللي المودل توقعها 


# In[66]:


x_test


# In[67]:


y_test


# In[68]:


model.intercept_


# In[69]:


model.coef_ 


# In[70]:


#statemodels


# In[71]:


import statsmodels.formula.api as sm


# In[72]:


lr=sm.ols(formula='area_mean ~ radius_mean +perimeter_mean',data=data[:600]).fit() 
# x=["radius_mean","perimeter_mean"]    y=["area_mean"] y~x #multilinear regression by formulaالطريقه التانيه لل


# In[73]:


#scaling


# In[74]:


#y=data["diagnosis"]
#x=data.radius_mean


# In[75]:


from sklearn.model_selection import train_test_split  #standarization


# In[76]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=.2)


# In[77]:


from sklearn.preprocessing import StandardScaler # mean , standard divission


# In[78]:


sc=StandardScaler()
x_train_scale=sc.fit_transform(x_train)
x_train_scale


# In[79]:


x_test_scale=sc.transform(x_test)
z=sc.inverse_transform(x_test_scale)
z


# In[80]:


from sklearn.preprocessing import MinMaxScaler # min max normalization


# In[81]:


MNS=MinMaxScaler()
x_train_MN=MNS.fit_transform(x_train) 
x_train_MN


# In[82]:


#classifiation


# In[83]:


from sklearn.neighbors import KNeighborsClassifier


# In[84]:


model=KNeighborsClassifier(n_neighbors=5) #numbers of neighbors where i select


# In[85]:


#model.fit(x_train_scale,y_train)


# In[ ]:


#y_predict=model.predict(x_train_scale)
#y_predict


# In[ ]:





# In[ ]:


#navie bayes classifier


# In[ ]:


from sklearn.naive_bayes import GaussianNB


# In[ ]:


GNB=GaussianNB()


# In[ ]:


#GNB.fit(x_train,y_train)


# In[ ]:


#y_predict=GNB.predict(x_test)
#y_predict


# In[ ]:





# In[ ]:


#svm


# In[ ]:


from sklearn import svm


# In[ ]:


classifer =svm.SVC(kernel='linear')


# In[ ]:


#classifer.fit(x_train,y_train)


# In[ ]:


#y_svm_predict=classifer.predict(x_test)
#y_svm_predict


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




