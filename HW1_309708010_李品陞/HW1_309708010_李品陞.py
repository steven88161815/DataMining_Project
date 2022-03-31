#!/usr/bin/env python
# coding: utf-8

# ## 資料前處理

# In[1]:


# 導入資料處理與可視化套件
get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# In[2]:


# 用pandas將資料讀進來並儲存成df
df = pd.read_csv('character-deaths.csv')
df


# In[3]:


# 看df的資料型態以及有無空值
df.info()


# In[4]:


# 看df的column有哪些
df.columns


# In[5]:


# 因為我們是要把空值轉成0(代表存活)，有數值的轉成1(代表死亡)，想確認我們想要預測的column們會不會發生本身非空但值就是0這種tricky的情況
ALL_Y = ['Death Year', 'Book of Death', 'Death Chapter']
for i in ALL_Y:
    print(df[i][df[i] == 0], end='\n\n')
# 發現在'Death Chapter'這一行有本身不是空值但值是0的角色，應該意味著這個角色沒有死，所以可以照著原本的做


# In[6]:


# 將空值補 0
df = df.fillna(0)


# In[7]:


# 再看一下df的資料型態以及有無空值，發現已無空值
df.info()


# In[8]:


# 選擇'Book of Death'作為我們想要預測的column，並儲存成Y
Y = df['Book of Death']
Y


# In[9]:


# 看我們的Y值有哪些列值不為0，並存到Y1
Y1 = Y[Y != 0]
Y1


# In[10]:


# 看一下Y1的index
Y1.index


# In[11]:


# 將這些列的值不為0的都替換成1
for i in Y1.index:
    Y.loc[i] = 1


# In[12]:


# Y的值變成只有0或1
Y


# In[13]:


# 取出我們想要用來預測Y的X們並儲存到X1，但因為'Allegiances'這一項還要做一些處理所以等等再放進來
X1 = df.iloc[:, 5:]
X1


# In[14]:


# 將'Allegiances'這一項轉變成one-hot-encoding的形式
X2 = pd.get_dummies(df['Allegiances'], prefix='Allegiances')
X2


# In[15]:


# 將X1和X2做合併成X，作為我們預測Y的參數
X = pd.concat([X1, X2], axis=1)
X


# ## 建模開始

# In[16]:


# 導入機器學習相關套件
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score
from collections import Counter


# In[17]:


# 將資料亂數拆成訓練集(75%)與測試集(25%) 
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=87)


# In[18]:


# 看一下資料的長度有沒有出錯
print(len(X_train))
print(len(X_test))
print(len(Y_train))
print(len(Y_test))


# ```
# 每次决策树分叉时，所有的特征都是随机排序的，随机种子就是random_state
# 如果你的max_features小于你总特征数n_features，那么每个分叉必须采样，随机性很大。
# 即使你的max_features = n_features，表现相同的分叉还是会选第一个，所以依然有随机性，
# sklearn的算法大多有random_state，如果需要复盘或是需要模型稳定不变必须设置。
# ```

# In[19]:


# 呼叫決策樹模型並限定深度為5，接著用訓練集來訓練
dtc = tree.DecisionTreeClassifier(max_depth=5, random_state=9487)
dtc.fit(X_train, Y_train)


# In[20]:


# 用訓練好的模型預測
Y_pred = dtc.predict(X_test)
Y_pred


# In[21]:


# 看一下預測出來為0和為1的數目
Counter(Y_pred)


# In[22]:


# 輸出混淆矩陣
confusion_matrix(Y_test, Y_pred, labels=[1, 0])


# In[23]:


precision_score(Y_test, Y_pred)


# In[24]:


recall_score(Y_test, Y_pred)


# In[25]:


accuracy_score(Y_test, Y_pred)


# ## 畫出決策樹的圖形

# In[26]:


# 导入决策树可视化套件
import graphviz
import pydotplus


# In[27]:


# 以DOT格式导出决策树
dot_data = tree.export_graphviz(dtc, out_file=None)


# In[ ]:





# In[28]:


# 第1種產生決策樹的方法


# In[29]:


graph = pydotplus.graph_from_dot_data(dot_data)


# In[30]:


graph.write_pdf("graph.pdf")


# In[ ]:





# In[31]:


# 第2種產生決策樹的方法


# In[32]:


graph = graphviz.Source(dot_data)
graph


# In[33]:


graph.render("decision_tree") 


# In[ ]:




