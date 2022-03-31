#!/usr/bin/env python
# coding: utf-8

# In[1]:


# 引入相關套件
get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter


# In[2]:


# 用pandas導入訓練集，分割符號採用table的方式，最後儲存成df1
df1 = pd.read_csv('train.csv', sep='\t') 
df1


# ## 前處理的方式跟HW4類似，但由於HW4模型準確率表現不夠好，試著對前處理做一些調整

# In[3]:


# 看一下df1的columns有哪些
df1.columns 


# In[4]:


# 看一下df1的index有哪些
df1.index 


# In[5]:


# 發現df1某一個label出錯
df1["label"].value_counts()


# In[6]:


# 看一下哪一列為True
df1["label"] == 'label'


# In[7]:


# 發現錯誤的列
df1[df1["label"] == 'label']


# In[8]:


# 刪掉該行
df1 = df1.drop([1615], axis=0)


# In[9]:


# 確認df1的label還有無出錯，發現已恢復正常
df1["label"].value_counts()


# In[10]:


# 更新一下df1的參數並看一下其樣子是否正確
df1.index = range(1, 4987)
df1


# In[11]:


# 引入自然語言處理套件
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

# 導入文字轉換成向量套件
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer # TfidfVectorizer為前兩者之混合，本程式碼因為中間需要拆開因此不會用到


# In[12]:


# 看一下隨機一個文本的樣子
ran = np.random.randint(4986)
df1.iloc[ran, 0]


# In[13]:


# 將所有文本放入all_words_train這個串列內，等等要做詞頻處理
all_words_train = []
for i in range(len(df1)):
    all_words_train.append(df1.iloc[i, 0])
    
all_words_train


# In[14]:


# 確認一下all_words_train的長度
len(all_words_train)


# In[15]:


# 使用詞頻矩陣套件並儲存成CV，超參數稍微改一下，跟HW4比更嚴格：token最少出現在20個文本中、但不能出現在1/4的文本以上以及stop_words設為英文
CV = CountVectorizer(max_df=0.25, min_df=20, stop_words='english')

# 將文本all_words_train帶入套件CV中
CV.fit(all_words_train)


# In[16]:


# 看一下總feature數
fn = CV.get_feature_names()
len(fn)


# In[17]:


# fit好的模型將all_words_train轉換
X1 = CV.transform(all_words_train)
X1


# In[18]:


# 確認一下矩陣的大小
X1.toarray().shape


# In[19]:


# 但我們真正要取的是tf-idf，因為他比較全面，所以先使用這個套件部儲存成tfidf
tfidf = TfidfTransformer()
# 將前面得到的轉換出來的結果帶入tfidf去fit和transform
X2 = tfidf.fit_transform(X1)


# In[20]:


# 看一下最終轉換出來的矩陣大小和模樣
matrix1 = X2.toarray()
print(matrix1.shape)
print(matrix1)


# In[21]:


# 將matrix1儲存成X_train
X_train = matrix1


# In[22]:


# 將訓練集的label取出來並儲存成y_train
y_train = df1['label']
y_train


# In[23]:


# 查看一下y_train的資料型態，發現是字串型態
y_train.dtype


# In[24]:


# 將其轉換成整數的資料型態
y_train = y_train.astype('int')
y_train.dtype


# ## 因為維度過高，因此先用PCA降維看看

# In[25]:


# 導入PCA套件
from sklearn.decomposition import PCA
pca = PCA(n_components=0.9)
pca.fit(X_train)


# In[26]:


# 做降維轉換
X_train_pca = pca.transform(X_train)


# In[27]:


# 看一下降維之後的形狀
X_train_pca.shape


# ## 再來是對測試集進行處理

# In[28]:


# 用pandas導入測試集，分割符號採用table的方式，最後儲存成df2
df2 = pd.read_csv('test.csv', sep='\t') 
df2


# In[29]:


# 將所有文本放入all_words_test這個串列內，等等要做詞頻處理
all_words_test = []
for i in range(len(df2)):
    all_words_test.append(df2.iloc[i, 1])
    
all_words_test


# In[30]:


# 確認一下all_words_test的長度
len(all_words_test)


# In[31]:


# fit好的模型將all_words_test轉換
X3 = CV.transform(all_words_test)
X3


# In[32]:


# 確認一下矩陣的大小
X3.toarray().shape


# In[33]:


# 將前面得到的轉換出來的結果帶入tfidf去fit和transform
X4 = tfidf.fit_transform(X3)


# In[34]:


# 看一下最終轉換出來的矩陣大小和模樣
matrix2 = X4.toarray()
print(matrix2.shape)
print(matrix2)


# In[35]:


# 將matrix2儲存成X_test
X_test = matrix2


# In[36]:


# 然後將前面做的pca套入X_test
X_test_pca = pca.transform(X_test)


# In[37]:


# 看一下降維之後的形狀
X_test_pca.shape


# ## 導入測試集的label

# In[38]:


# 將含有測試集label的csv檔導入
df3 = pd.read_csv('sample_submission (1).csv') 
df3


# In[39]:


# 將label儲存成y_test
y_test = df3.iloc[:, 1]
y_test


# In[40]:


# 將y_test轉變成array的形式
y_test = y_test.values
y_test


# In[41]:


# 查看一下y_test的資料型態，發現是整數，不需做處理
y_test.dtype


# ## 使用RNN與LSTM進行建模

# ## 建模_RNN

# In[42]:


# 引入相關套件
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import SimpleRNN


# In[43]:


# 建立模型
modelRNN = Sequential()

# Embedding層將「數字list」轉換成「向量list」
# 輸出的維度是32，希望將數字list轉換為32維度的向量
# 輸入的維度是2142，也就是我們之前建立的字典是1969字
# 數字list截長補短後都是200個數字
modelRNN.add(Embedding(output_dim=32,   
             input_dim=2142,            
             input_length=200))         

# 加入Dropout，避免overfitting
# 隨機在神經網路中放棄20%的神經元，避免overfitting
modelRNN.add(Dropout(0.2)) 


# In[44]:


# 建立RNN層
# 建立16個神經元的RNN層
modelRNN.add(SimpleRNN(units=16)) 

# 建立隱藏層
# 建立256個神經元的隱藏層 
# ReLU激活函數
modelRNN.add(Dense(units=256, activation='relu'))  
modelRNN.add(Dropout(0.7))

# 建立輸出層
# 建立一個神經元的輸出層 
# Sigmoid激活函數
modelRNN.add(Dense(units=1, activation='sigmoid')) 


# In[45]:


# 查看模型摘要
modelRNN.summary()


# In[46]:


# 定義訓練模型
# Loss function使用Cross entropy
# adam最優化方法可以更快收斂
modelRNN.compile(loss='binary_crossentropy',
                 optimizer='adam',
                 metrics=['accuracy']) 


# In[47]:


# validation_split=0.2 設定80%訓練資料、20%驗證資料
# 執行10次訓練週期
# 每一批次訓練100筆資料
# verbose 顯示訓練過程
train_history1 = modelRNN.fit(X_train_pca, y_train, 
                             epochs=10, 
                             batch_size=100,
                             verbose=1,
                             validation_split=0.2)


# In[48]:


# 看一下此RNN模型迭代過程中'loss','accuracy','val_loss','val_accuracy'的變化
train_history1.history


# In[49]:


def show_train_history_Acc(train_history):
    plt.figure()
    plt.plot(train_history.history['accuracy'])
    plt.plot(train_history.history['val_accuracy'])
    plt.title("Train History")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend(["train", "validation"], loc="upper left")
    plt.show()


# In[50]:


show_train_history_Acc(train_history1)


# In[51]:


def show_train_history_Loss(train_history):
    plt.figure()
    plt.plot(train_history.history['loss'])
    plt.plot(train_history.history['val_loss'])
    plt.title("Train History")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend(["train", "validation"], loc="upper left")
    plt.show()


# In[52]:


show_train_history_Loss(train_history1) 


# In[53]:


# 使用test測試資料及評估準確率
# 1247/32大約等於38.96
scores1 = modelRNN.evaluate(X_test_pca, y_test, verbose=1)
scores1


# Keras中model.evaluate()：返回的是损失值和你选定的指标值（例如，精度accuracy)

# In[54]:


# 因此我們取第二個值
scores1[1]


# ## 建模_LSTM

# In[55]:


# 匯入相關套件
from keras.models import Sequential
from keras.layers.core import Dense,Dropout,Activation,Flatten
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM


# In[56]:


# 建立模型
modelLSTM = Sequential() 
# Embedding層將「數字list」轉換成「向量list」
# 輸出的維度是32，希望將數字list轉換為32維度的向量
# 輸入的維度是2142，也就是我們之前建立的字典是1969字
# 數字list截長補短後都是200個數字
modelLSTM.add(Embedding(output_dim=32,       
                        input_dim=2142,    
                        input_length=200))         

# 加入Dropout，避免overfitting
# 隨機在神經網路中放棄20%的神經元，避免overfitting
modelLSTM.add(Dropout(0.2)) 


# In[57]:


# 建立LSTM層
# 建立32個神經元的LSTM層
modelLSTM.add(LSTM(32)) 


# 建立隱藏層
# 建立256個神經元的隱藏層
modelLSTM.add(Dense(units=256,activation='relu')) 
modelLSTM.add(Dropout(0.7))

# 建立輸出層
# 建立一個神經元的輸出層
modelLSTM.add(Dense(units=1,activation='sigmoid'))


# In[58]:


# 查看模型摘要
modelLSTM.summary()


# In[59]:


# 定義訓練模型
# Loss function使用Cross entropy
# adam最優化方法可以更快收斂
modelLSTM.compile(loss='binary_crossentropy',
                 optimizer='adam',
                 metrics=['accuracy']) 


# In[60]:


# validation_split=0.2 設定80%訓練資料、20%驗證資料
# 執行10次訓練週期
# 每一批次訓練200筆資料
# verbose 顯示訓練過程
train_history2 = modelLSTM.fit(X_train_pca, y_train, 
                             epochs=10, 
                             batch_size=100,
                             verbose=1,
                             validation_split=0.2)


# In[61]:


# 看一下此LSTM模型迭代過程中'loss','accuracy','val_loss','val_accuracy'的變化
train_history2.history


# In[62]:


show_train_history_Acc(train_history2)


# In[63]:


show_train_history_Loss(train_history2) 


# In[64]:


# 使用test測試資料及評估準確率
scores2 = modelLSTM.evaluate(X_test_pca, y_test, verbose=1)
scores2


# In[65]:


# 取出accuracy
scores2[1]


# In[ ]:




