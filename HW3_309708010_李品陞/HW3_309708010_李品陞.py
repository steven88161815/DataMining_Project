#!/usr/bin/env python
# coding: utf-8

# ## 資料前處理

# In[1]:


# 導入套件
get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from collections import Counter


# In[2]:


# 忽略警告
import warnings
warnings.filterwarnings("ignore")


# In[3]:


# 讀入作業的csv檔並存到df
df = pd.read_csv('新竹_2019.csv', encoding = 'ANSI')
df


# In[4]:


# 查看df的資料型態以及有無空值
df.info()


# In[5]:


# 看一下第二行第二列的值，發現結尾有空白
df.iloc[1, 1]


# In[6]:


# 看一下df的column，發現也一堆空白
df.columns 


# In[7]:


# 寫迴圈創造一群其字首字尾都沒有空白的columns取代原本的columns(也就是把最外層的空白去掉)
col1 = []
for i in df.columns:
    k = i.strip()
    col1.append(k)
print(col1)

df.columns = col1


# In[8]:


# 重新檢查df的column發現已恢復正常
df.columns


# In[9]:


# 看一下df的測項這一行
df['測項']


# In[10]:


# 隨機取一列發現也有結尾空白的現象
df['測項'][2]


# In[11]:


# 利用迴圈與strip()把df裡的值的多餘空白都去掉
for i in range(6481):
    for j in range(27):
        df.iloc[i, j] = df.iloc[i, j].strip()


# In[12]:


# 再檢查一次發現已恢復正常
df['測項'][2]


# In[13]:


# 看一下df的日期這一行
df['日期']


# In[14]:


# 將日期為十月到十二月的數據取出，並看一下長度
df0 = df[(df['日期'] >= '2019/10/01 00:00:00') & (df['日期'] <= '2019/12/31 00:00:00')]
len(df0)


# In[15]:


# 更新指標
df0.index = [i for i in range(1, 1657)]
df0


# In[16]:


# NR表示無降雨，以0取代
df0 = df0.replace('NR', '0')


# In[17]:


# 缺失值以及無效值以前後一小時平均值取代 (如果前一小時仍有空值，再取更前一小時)
# 只需關注column為'00'~'23'的行，所以j才會這麼取
for i in range(1656):
    for j in range(3, 27):
        m = j-1
        if m == 2:
            m = 26
        n = j+1
        if n == 27:
            n = 3
        
        if ((df0.iloc[i, j].find('#') != -1) | (df0.iloc[i, j].find('*') != -1) | (df0.iloc[i, j].find('x') != -1) | (df0.iloc[i, j].find('A') != -1) | (np.isnan(df0.iloc[i, j] == True))):
            while ((df0.iloc[i, m].find('#') != -1) | (df0.iloc[i, m].find('*') != -1) | (df0.iloc[i, m].find('x') != -1) | (df0.iloc[i, m].find('A') != -1) | (np.isnan(df0.iloc[i, m] == True))):
                m = m-1
                if m == 2:
                    m = 26
            while ((df0.iloc[i, n].find('#') != -1) | (df0.iloc[i, n].find('*') != -1) | (df0.iloc[i, n].find('x') != -1) | (df0.iloc[i, n].find('A') != -1) | (np.isnan(df0.iloc[i, n] == True))):
                n = n+1
                if n == 27:
                    n = 3
                    
            df0.iloc[i, j] = str((float(df0.iloc[i, m]) + float(df0.iloc[i, n]))/2)            


# In[18]:


# 檢查一下有無漏網之魚，沒有就會印出0
a = 0
for i in range(1656):
    for j in range(3, 27):
        if (df0.iloc[i, j].find('#') != -1) | (df0.iloc[i, j].find('*') != -1) | (df0.iloc[i, j].find('x') != -1) | (df0.iloc[i, j].find('A') != -1) | (np.isnan(df0.iloc[i, j] == True)):
            a = 1

print(a)


# In[19]:


# 將column為'00'~'23'的行取出並存到col2
col2 = []
for i in range(3, 27):
    col2.append(col1[i])
# 印出col2
print(col2)


# In[20]:


# 更改df0內column為'00'~'23'值的資料型態(從str改成float)
for i in col2:
    df0[i] = df0[i].astype(float)


# In[21]:


# 觀看df0的資料型態，發現已改
df0.info()


# In[22]:


# 將df0內的十月十一月當訓練集儲存成train，十二月當測試集儲存成test
train = df0[(df0['日期'] >= '2019/10/01 00:00:00') & (df0['日期'] <= '2019/11/30 00:00:00')]
test = df0[(df0['日期'] >= '2019/12/01 00:00:00') & (df0['日期'] <= '2019/12/31 00:00:00')]


# In[23]:


# 看一下train的樣子
train


# In[24]:


# 因為等等要將訓練集每18列作合併，避免搞混測項，先將測項當作第一行
A = train.iloc[0:18, 2:3]
A


# In[25]:


# 用迴圈將訓練集每18列作合併，要注意的是每次取出的18列要將指標更新成1~18，不然合併會出問題
for i in range(int(len(train)/18)):
    B = train.iloc[18*i:18*(i+1), 3:27]
    B.index = [j for j in range(1, 19)]
    A = pd.concat([A, B], axis=1)
# 輸出合併後的結果
A
# 先記住PM2.5那一列的指標是10(第10列)


# In[26]:


# 將A的測項那一行去掉，然後用train儲存
train = A.iloc[:, 1:]
train


# In[27]:


# 將train從dataframe轉變成array的形式，更方便做相關操作
train = train.values
# 查看陣列型態的train的模樣及形狀
print(train)
print(train.shape)


# In[28]:


# 先看一下PM2.5那列長什麼樣子
train[9]


# ### 先做只有取PM2.5的X

# #### 取未來第一個小時當作預測目標

# In[29]:


# 取出PM2.5那一列，然後用儲存成PM
# 然後產生1458個X跟Y用於訓練
PM = train[9]
X1_train = []
Y1_train = []
for i in range(len(PM)-6):
    box = []
    for j in range(6):
        box.append(PM[j])
    X1_train.append(box)
    Y1_train.append(PM[6])
    PM = PM[1:]


# In[30]:


# 先轉成array，然後再查看其模樣
X1_train = np.array(X1_train)
print(X1_train)
print(X1_train.shape)

# Y1_train做法一樣
Y1_train = np.array(Y1_train)
print(Y1_train)
print(Y1_train.shape)


# #### 取未來第六個小時當作預測目標

# In[31]:


# 如法炮製做未來第六個小時的
PM = train[9]
X2_train = []
Y2_train = []
for i in range(len(PM)-11):
    box = []
    for j in range(6):
        box.append(PM[j])
    X2_train.append(box)
    Y2_train.append(PM[11])
    PM = PM[1:]


# In[32]:


X2_train = np.array(X2_train)
print(X2_train)
print(X2_train.shape)

Y2_train = np.array(Y2_train)
print(Y2_train)
print(Y2_train.shape)


# ### 再來取全部的當作X

# In[33]:


# 先回顧一下train的模樣
train


# In[34]:


# train的形狀
train.shape


# #### 取未來第一個小時當作預測目標

# In[35]:


# 為了避免等等使用train被覆蓋，先產生一個train的複製品叫cotrain，方便重複使用
cotrain = train


# In[36]:


# 作法與上方雷同，比較麻煩的是X3_train的取值變多了，需要多一層迴圈
X3_train = []
Y3_train = []
for i in range(len(cotrain[9])-6):
    box = []
    for j in range(18):
        for k in range(6):
            box.append(cotrain[j, k])
    X3_train.append(box)
    Y3_train.append(cotrain[9, 6])
    cotrain = cotrain[:, 1:]


# In[37]:


X3_train = np.array(X3_train)
print(X3_train)
print(X3_train.shape)

Y3_train = np.array(Y3_train)
print(Y3_train)
print(Y3_train.shape)


# In[38]:


# 檢驗一下Y3_train有沒有等於Y1_train(不同代表算錯了)，發現沒算錯
Counter(Y1_train == Y3_train)


# #### 取未來第六個小時當作預測目標

# In[39]:


cotrain = train


# In[40]:


X4_train = []
Y4_train = []
for i in range(len(cotrain[9])-11):
    box = []
    for j in range(18):
        for k in range(6):
            box.append(cotrain[j, k])
    X4_train.append(box)
    Y4_train.append(cotrain[9, 11])
    cotrain = cotrain[:, 1:]


# In[41]:


X4_train = np.array(X4_train)
print(X4_train)
print(X4_train.shape)

Y4_train = np.array(Y4_train)
print(Y4_train)
print(Y4_train.shape)


# In[42]:


# 檢驗一下Y4_train有沒有等於Y2_train(不同代表算錯了)，發現沒算錯
Counter(Y4_train == Y2_train)


# ### train的部分做完了，接下來做test的部分，動作幾乎一模一樣

# In[43]:


# 看一下test長的樣子
test


# In[44]:


# 更新指標
test.index = [i for i in range(1, 559)]
test


# In[45]:


A = test.iloc[0:18, 2:3]
for i in range(int(len(test)/18)):
    B = test.iloc[18*i:18*(i+1), 3:27]
    B.index = [j for j in range(1, 19)]
    A = pd.concat([A, B], axis=1)
A


# In[46]:


test = A.iloc[:, 1:]

test = test.values
print(test)
print(test.shape)


# ### 先做只有取PM2.5的X

# #### 取未來第一個小時當作預測目標

# In[47]:


PM = test[9]
X1_test = []
Y1_test = []
for i in range(len(PM)-6):
    box = []
    for j in range(6):
        box.append(PM[j])
    X1_test.append(box)
    Y1_test.append(PM[6])
    PM = PM[1:]


# In[48]:


X1_test = np.array(X1_test)
print(X1_test)
print(X1_test.shape)
Y1_test = np.array(Y1_test)


# #### 取未來第六個小時當作預測目標

# In[49]:


PM = test[9]
X2_test = []
Y2_test = []
for i in range(len(PM)-11):
    box = []
    for j in range(6):
        box.append(PM[j])
    X2_test.append(box)
    Y2_test.append(PM[11])
    PM = PM[1:]


# In[50]:


X2_test = np.array(X2_test)
print(X2_test)
print(X2_test.shape)
Y2_test = np.array(Y2_test)


# ### 再來取全部的當作X

# #### 取未來第一個小時當作預測目標

# In[51]:


# 為了避免等等使用test被覆蓋，先產生一個test的複製品叫cotest，方便重複使用
cotest = test


# In[52]:


X3_test = []
Y3_test = []
for i in range(len(cotest[9])-6):
    box = []
    for j in range(18):
        for k in range(6):
            box.append(cotest[j, k])
    X3_test.append(box)
    Y3_test.append(cotest[9, 6])
    cotest = cotest[:, 1:]


# In[53]:


X3_test = np.array(X3_test)
print(X3_test)
print(X3_test.shape)
Y3_test = np.array(Y3_test)


# In[54]:


# 檢驗一下Y3_test有沒有等於Y1_test(不同代表算錯了)，發現沒算錯
Counter(Y1_test == Y3_test)


# #### 取未來第六個小時當作預測目標

# In[55]:


cotest = test


# In[56]:


X4_test = []
Y4_test = []
for i in range(len(cotest[9])-11):
    box = []
    for j in range(18):
        for k in range(6):
            box.append(cotest[j, k])
    X4_test.append(box)
    Y4_test.append(cotest[9, 11])
    cotest = cotest[:, 1:]


# In[57]:


X4_test = np.array(X4_test)
print(X4_test)
print(X4_test.shape)
Y4_test = np.array(Y4_test)


# In[58]:


# 檢驗一下Y4_test有沒有等於Y2_test(不同代表算錯了)，發現沒算錯
Counter(Y2_test == Y4_test)


# ### 到這邊前處理正式結束，真的非常複雜！

# In[ ]:





# ## 線性回歸建模

# In[59]:


from sklearn.linear_model import LinearRegression
regr = LinearRegression()


# ### 第一種

# In[60]:


regr.fit(X1_train, Y1_train)
Y1_predict = regr.predict(X1_test)


# In[61]:


MAE_Y1_0 = 0
for i in (Y1_predict-Y1_test):
    MAE_Y1_0 += abs(i)
MAE_Y1_0 = MAE_Y1_0/len(Y1_predict)
MAE_Y1_0


# ### 第二種

# In[62]:


regr.fit(X2_train, Y2_train)
Y2_predict = regr.predict(X2_test)


# In[63]:


MAE_Y2_0 = 0
for i in (Y2_predict-Y2_test):
    MAE_Y2_0 += abs(i)
MAE_Y2_0 = MAE_Y2_0/len(Y2_predict)
MAE_Y2_0


# ### 第三種

# In[64]:


regr.fit(X3_train, Y3_train)
Y3_predict = regr.predict(X3_test)


# In[65]:


MAE_Y3_0 = 0
for i in (Y3_predict-Y3_test):
    MAE_Y3_0 += abs(i)
MAE_Y3_0 = MAE_Y3_0/len(Y3_predict)
MAE_Y3_0


# ### 第四種

# In[66]:


regr.fit(X4_train, Y4_train)
Y4_predict = regr.predict(X4_test)


# In[67]:


MAE_Y4_0 = 0
for i in (Y4_predict-Y4_test):
    MAE_Y4_0 += abs(i)
MAE_Y4_0 = MAE_Y4_0/len(Y4_predict)
MAE_Y4_0


# In[ ]:





# ## 隨機森林建模

# In[68]:


from sklearn.ensemble import RandomForestRegressor


# In[69]:


# 回歸森林：樹木設100棵、最大深度設為8、因為作業是問mae所以設定criterion='mae'
regr = RandomForestRegressor(n_estimators=100, max_depth=8, criterion='mae', random_state=9487)


# ### 第一種

# In[70]:


regr.fit(X1_train, Y1_train)
Y1_predict = regr.predict(X1_test)


# In[71]:


MAE_Y1_1 = 0
for i in (Y1_predict-Y1_test):
    MAE_Y1_1 += abs(i)
MAE_Y1_1 = MAE_Y1_1/len(Y1_predict)
MAE_Y1_1


# ### 第二種

# In[72]:


regr.fit(X2_train, Y2_train)
Y2_predict = regr.predict(X2_test)


# In[73]:


MAE_Y2_1 = 0
for i in (Y2_predict-Y2_test):
    MAE_Y2_1 += abs(i)
MAE_Y2_1 = MAE_Y2_1/len(Y2_predict)
MAE_Y2_1


# ### 第三種

# In[74]:


regr.fit(X3_train, Y3_train)
Y3_predict = regr.predict(X3_test)


# In[75]:


MAE_Y3_1 = 0
for i in (Y3_predict-Y3_test):
    MAE_Y3_1 += abs(i)
MAE_Y3_1 = MAE_Y3_1/len(Y3_predict)
MAE_Y3_1


# ### 第四種

# In[76]:


regr.fit(X4_train, Y4_train)
Y4_predict = regr.predict(X4_test)


# In[77]:


MAE_Y4_1 = 0
for i in (Y4_predict-Y4_test):
    MAE_Y4_1 += abs(i)
MAE_Y4_1 = MAE_Y4_1/len(Y4_predict)
MAE_Y4_1


# In[ ]:





# ## 總結

# In[78]:


print(f'只有取PM2.5當作X，將未來第一個小時PM2.5當預測目標，用線性回歸建模，其MAE為{MAE_Y1_0:.2f}')
print(f'只有取PM2.5當作X，將未來第六個小時PM2.5當預測目標，用線性回歸建模，其MAE為{MAE_Y2_0:.2f}')
print(f'全部汙染物當作X，將未來第一個小時PM2.5當預測目標，用線性回歸建模，其MAE為{MAE_Y3_0:.2f}')
print(f'全部汙染物當作X，將未來第六個小時PM2.5當預測目標，用線性回歸建模，其MAE為{MAE_Y4_0:.2f}')
print(f'只有取PM2.5當作X，將未來第一個小時PM2.5當預測目標，用隨機森林建模，其MAE為{MAE_Y1_1:.2f}')
print(f'只有取PM2.5當作X，將未來第六個小時PM2.5當預測目標，用隨機森林建模，其MAE為{MAE_Y2_1:.2f}')
print(f'全部汙染物當作X，將未來第一個小時PM2.5當預測目標，用隨機森林建模，其MAE為{MAE_Y3_1:.2f}')
print(f'全部汙染物當作X，將未來第六個小時PM2.5當預測目標，用隨機森林建模，其MAE為{MAE_Y4_1:.2f}')


# In[ ]:




