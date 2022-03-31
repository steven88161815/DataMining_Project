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

## 導入文字轉換成向量套件
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


# 使用詞頻矩陣套件並儲存成CV，設定超參數：token最少出現在兩個文本中、但不能出現超過總文本的一半以及stop_words設為英文
CV = CountVectorizer(max_df=0.5, min_df=2, stop_words='english')
# 將文本all_words_train帶入套件CV中
CV.fit(all_words_train)


# In[16]:


# 看一下總feature數
fn = CV.get_feature_names()
len(fn)


# In[17]:


# fit好的模型將all_words_train做轉換
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


# ## 因為維度過高，因此先用PCA降維看看

# In[23]:


# 導入PCA套件
from sklearn.decomposition import PCA
pca = PCA(n_components=0.8)
pca.fit(X_train)


# In[24]:


# 做降維轉換
X_train_pca = pca.transform(X_train)


# In[25]:


# 看一下降維之後的形狀
X_train_pca.shape


# ## 再來是對測試集進行處理

# In[26]:


# 用pandas導入測試集，分割符號採用table的方式，最後儲存成df2
df2 = pd.read_csv('test.csv', sep='\t') 
df2


# In[27]:


# 將所有文本放入all_words_test這個串列內，等等要做詞頻處理
all_words_test = []
for i in range(len(df2)):
    all_words_test.append(df2.iloc[i, 1])
    
all_words_test


# In[28]:


# 確認一下all_words_test的長度
len(all_words_test)


# In[29]:


# fit好的模型將all_words_test做轉換
X3 = CV.transform(all_words_test)
X3


# In[30]:


# 確認一下矩陣的大小
X3.toarray().shape


# In[31]:


# 將前面得到的轉換出來的結果帶入tfidf去fit和transform
X4 = tfidf.fit_transform(X3)


# In[32]:


# 看一下最終轉換出來的矩陣大小和模樣
matrix2 = X4.toarray()
print(matrix2.shape)
print(matrix2)


# In[33]:


# 將matrix2儲存成X_test
X_test = matrix2


# In[34]:


# 然後將前面做的pca套入X_test
X_test_pca = pca.transform(X_test)


# In[35]:


# 看一下降維之後的形狀
X_test_pca.shape


# ## 導入測試集的label

# In[36]:


# 將含有測試集label的csv檔導入
df3 = pd.read_csv('sample_submission.csv') 
df3


# In[37]:


# 將label儲存成y_test
y_test = df3.iloc[:, 1]
y_test


# In[38]:


# 將y_test轉變成array的形式
y_test = y_test.values
y_test


# In[ ]:





# ## 建模

# In[39]:


# 建模前先引入相關套件
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score


# In[ ]:





# ## 建模_xgboost

# In[40]:


# 引入xgboost套件並儲存成xgb
import xgboost as xgb


# In[41]:


dtrain=xgb.DMatrix(X_train_pca, label=y_train)
dtest=xgb.DMatrix(X_test_pca)


# In[42]:


# 調整超參數
params={'booster':'gbtree',
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'max_depth':4,
        'lambda':10,
        'subsample':0.75,
        'colsample_bytree':0.75,
        'min_child_weight':2,
        'eta': 0.05,
        'seed':0,
        }


# In[43]:


watchlist = [(dtrain,'train')]


# In[44]:


bst=xgb.train(params, dtrain, num_boost_round=5, evals=watchlist)


# In[45]:


# 输出概率
y_pred=bst.predict(dtest)
y_pred


# In[46]:


# 確認一下y_pred長度
len(y_pred)


# In[47]:


# 设置阈值, 输出一些评价指标，选择概率大于0.5的为1，其他为0类
y_pred = (y_pred >= 0.5)*1
y_pred


# In[48]:


print(f'accuracy_score是{accuracy_score(y_test, y_pred)}')
print(f'precision_score是{precision_score(y_test, y_pred)}')
print(f'recall_score是{recall_score(y_test, y_pred)}')
print(f'f1_score是{f1_score(y_test, y_pred)}')


# In[ ]:





# ## 建模_GBDT

# In[49]:


# 引入GradientBoostingClassifier套件
from sklearn.ensemble import GradientBoostingClassifier


# In[50]:


# 調整超參數
gbc = GradientBoostingClassifier(n_estimators=1000, max_features="sqrt", max_depth=8, random_state=0,                                 subsample=0.75, min_samples_split=2, learning_rate=0.1)


# In[51]:


# 擬合模型
gbc.fit(X_train_pca, y_train)


# In[52]:


# 看一下擬合完對原訓練集的預測準確率
accuracy_score(y_train, gbc.predict(X_train_pca))


# In[53]:


# 輸出預測的array，但發現裡面的內容竟然是string
y_pred = gbc.predict(X_test_pca)
y_pred


# In[54]:


# 將其資料型態改成整數
y_pred = y_pred.astype('int')
y_pred


# In[55]:


# 確認一下y_pred長度
len(y_pred)


# In[56]:


print(f'accuracy_score是{accuracy_score(y_test, y_pred)}')
print(f'precision_score是{precision_score(y_test, y_pred)}')
print(f'recall_score是{recall_score(y_test, y_pred)}')
print(f'f1_score是{f1_score(y_test, y_pred)}')


# In[ ]:





# ## 建模_LightGBM

# In[57]:


# 引入LightGBM套件
from lightgbm import LGBMClassifier


# In[58]:


# 調整超參數
LGBMC = LGBMClassifier(num_leaves=50, n_estimators=100, colsample_bytree=0.1, max_depth=5, random_state=87, subsample=0.75,                       learning_rate=0.1)


# In[59]:


# 擬合模型
LGBMC.fit(X_train_pca, y_train)


# In[60]:


# 看一下擬合完對原訓練集的預測準確率
accuracy_score(y_train, LGBMC.predict(X_train_pca))


# In[61]:


# 預測測試集的label並儲存成y_pred，但發現裡面的內容竟然是string
y_pred = LGBMC.predict(X_test_pca)
y_pred


# In[62]:


# 將裡面的資料型態改成整數
y_pred = y_pred.astype('int')
y_pred


# In[63]:


# 確認一下y_pred長度
len(y_pred)


# In[64]:


# 輸出預測結果
print(f'accuracy_score是{accuracy_score(y_test, y_pred)}')
print(f'precision_score是{precision_score(y_test, y_pred)}')
print(f'recall_score是{recall_score(y_test, y_pred)}')
print(f'f1_score是{f1_score(y_test, y_pred)}')


# In[ ]:




