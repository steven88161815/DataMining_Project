#!/usr/bin/env python
# coding: utf-8

# # 309708010_李品陞_資料探勘_HW1

# ## 鐵達尼號預測(16%)
# 參考資料：
# [資料分析&機器學習 第4.1講：Kaggle競賽-鐵達尼號生存預測(前16%排名)](https://medium.com/jameslearningnote/%E8%B3%87%E6%96%99%E5%88%86%E6%9E%90-%E6%A9%9F%E5%99%A8%E5%AD%B8%E7%BF%92-%E7%AC%AC4-1%E8%AC%9B-kaggle%E7%AB%B6%E8%B3%BD-%E9%90%B5%E9%81%94%E5%B0%BC%E8%99%9F%E7%94%9F%E5%AD%98%E9%A0%90%E6%B8%AC-%E5%89%8D16-%E6%8E%92%E5%90%8D-a8842fea7077)

# In[1]:


from sklearn import preprocessing # 導入數據預處理套件
from sklearn.model_selection import GridSearchCV # 導入尋找超參數的套件
from sklearn.ensemble import RandomForestClassifier # 導入隨機森林分類器
from sklearn.ensemble import RandomForestRegressor # 導入隨機森林迴歸器

# 導入資料處理與可視化套件
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

get_ipython().run_line_magic('matplotlib', 'inline')
pd.options.mode.chained_assignment = None # 忽略警告信息


# In[2]:


# 將csv檔導入並儲存到train test submit
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
submit = pd.read_csv('gender_submission.csv')


# In[3]:


print(train.shape)
print(test.shape)
print(submit.shape)


# In[4]:


train.head(5) # 觀察train的前五筆資料


# In[5]:


test.head(5) # 觀察test的前五筆資料


# In[6]:


train.dtypes # 觀察train資料型態


# In[7]:


train.info() # 觀察train資料型態以及有無空值


# In[8]:


test.info() # 觀察test資料型態以及有無空值


# In[9]:


train.describe() # 觀察train的資料分布


# In[10]:


test.describe() # 觀察test的資料分布


# ## Combine Train and Test Data

# In[11]:


data = train.append(test) # 將test與train做合併，也就是把test接在train的後面，然後儲存成data
data # 輸出data


# In[12]:


data.reset_index(inplace=True, drop=True) # 重置data的指標，drop=True代表新的index會覆蓋原本的index
data


# ## Data Analysis

# In[13]:


# countplot 計數圖
sns.countplot(data['Survived']) # 觀察生存人數


# In[14]:


sns.countplot(data.Pclass, hue=data['Survived']) # 觀察艙等跟生存率的關係


# In[15]:


sns.countplot(data['Pclass'], hue=data['Survived']) # 觀察艙等跟生存率的關係


# In[16]:


sns.countplot(data['Sex'], hue=data['Survived']) # 觀察性別跟生存率的關係


# In[17]:


sns.countplot(data['Embarked'], hue=data['Survived']) # 觀察出發港口跟生存率的差異


# In[18]:


# 觀察年齡跟生存率的關係，因為年齡是連續的數值，所以分成兩個以上的表格看
g = sns.FacetGrid(data, col='Survived')
g.map(sns.distplot, 'Age', kde=False)


# In[19]:


# 觀察票價跟生存率的關係，因為票價是連續的數值，所以分成兩個以上的表格看
g = sns.FacetGrid(data, col='Survived')
g.map(sns.distplot, 'Fare', kde=False)


# In[20]:


# 觀察父母+小孩的數量跟生存率的關係，因為父母+小孩的數量是離散的數值，所以也可以用剛剛使用過的countplot來看
g = sns.FacetGrid(data, col='Survived')
g.map(sns.distplot, 'Parch', kde=False)


# In[21]:


sns.countplot(data['Parch'], hue=data['Survived']) # 觀察父母+小孩的數量跟生存率的差異


# In[22]:


# 觀察兄弟姊妹＋配偶的數量的數量跟生存率的關係
g = sns.FacetGrid(data, col='Survived')
g.map(sns.distplot, 'SibSp', kde=False)


# In[23]:


data # 原本data的樣子


# In[24]:


data['Family_Size'] = data['Parch'] + data['SibSp'] # 成立一個家庭成員數量(不算自己)的欄位，是由父母+小孩的數量與兄弟姊妹＋配偶的數量


# In[25]:


data # 會發現多了一個 Family_Size


# In[26]:


# 觀察家庭成員數量的數量跟生存率的關係
g = sns.FacetGrid(data, col='Survived')
g.map(sns.distplot, 'Family_Size', kde=False)


# ## Feature Engineering

# In[27]:


data['Name'].str.split(", ", expand=False)


# In[28]:


data['Name'].str.split(", ", expand=True) # expand=True可以把用split分開的兩個輸出到dataframe不同的column


# In[29]:


data['Title1'] = data['Name'].str.split(", ", expand=True)[1] # 利用逗號分開名字和稱謂+後面的東東，並存進Title1
data['Title1'].head(3) # 輸出Title1的前三項


# In[30]:


data['Title1'] = data['Title1'].str.split(".", expand=True)[0] # 利用句號分開跟稱謂+後面的東西，將稱謂取出來，並存進Title1
data['Title1'].head(3) # 輸出Title1的前三項


# In[31]:


data['Title1'].unique() # 輸出有哪些稱謂


# In[32]:


pd.crosstab(data['Title1'], data['Sex']).T.style.background_gradient(cmap='summer_r') # 輸出稱謂與性別的交叉表


# In[33]:


pd.crosstab(data['Title1'], data['Survived']).T.style.background_gradient(cmap='summer_r') # 輸出稱謂與存活與否的交叉表


# In[34]:


data.groupby(['Title1'])['Age'].mean() # 輸出各個稱謂的平均壽命


# In[35]:


data.groupby(['Title1','Pclass'])['Age'].mean() # 輸出各個稱謂在不同艙等下的平均壽命


# In[36]:


data['Title2'] = data['Title1'].replace(['Mlle','Mme','Ms','Dr','Major','Lady','the Countess','Jonkheer','Col','Rev',                                         'Capt','Sir','Don','Dona'], ['Miss','Mrs','Miss','Mr','Mr','Mrs','Mrs','Mr',                                                                      'Mr','Mr','Mr','Mr','Mr','Mrs'])
# 若仔細觀察這些稱謂，會發現有些是稱謂的乘客非常少，因此我們把其中的稱謂做合併，然後存到Title2。


# In[37]:


data.groupby('Title2')['Age'].mean() # 輸出四個稱謂的平均壽命


# In[38]:


data.groupby(['Title2','Pclass'])['Age'].mean() # 輸出四個稱謂在不同艙等下的平均壽命


# In[39]:


pd.crosstab(data['Title2'],data['Sex']).T.style.background_gradient(cmap='summer_r') # 輸出這四種稱謂和性別的交叉表


# In[40]:


pd.crosstab(data['Title2'],data['Survived']).T.style.background_gradient(cmap='summer_r') # 輸出這四種稱謂和存活與否的交叉表


# In[41]:


data.groupby(['Title2','Pclass'])['Age'].mean().iteritems()


# In[42]:


list(data.groupby(['Title2','Pclass'])['Age'].mean().iteritems())[:3] # 列出這四種稱謂在不同艙等下的平均壽命的前三項


# In[43]:


data['Ticket'] # 看船票的信息


# In[44]:


# 再來把票號的資訊取出前面英文的部分，因為相同的英文代碼可能代表的是房間的位置，後面的號碼沒有意義所以省略，如果只有號碼的票號就用X來表示
# 接著把結果存進Ticket_info這個新欄位
data['Ticket_info'] = data['Ticket'].apply(lambda x : x.replace(".","").replace("/","").strip().split(' ')[0]                                            if not x.isdigit() else 'X')


# In[45]:


data['Ticket_info'].unique() # 輸出Ticket_info的種類


# In[46]:


sns.countplot(data['Ticket_info'], hue=data['Survived']) # 輸出不同種類Ticket_info的與生存率的關係


# ## Missing Value-embarked、Fare、Age

# In[47]:


data.groupby(['Embarked']).count() # 看各個登船港口的人數


# In[48]:


data.info() # 觀察data的資料型態以及有無空值


# In[49]:


data['Embarked'] = data['Embarked'].fillna('S') # 由於登船港口(Embarked)只有遺漏少數，我們就直接補上出現次數最多的 "S"


# In[50]:


data.info() # 發現Embarked已無空值


# In[51]:


data['Fare'] = data['Fare'].fillna(data['Fare'].mean()) # 用平均值補票價的空值


# In[52]:


data.info() # 發現Fare已無空值


# In[53]:


data['Cabin'].head(10) # 看客艙的前10筆資料


# In[54]:


data["Cabin"] = data['Cabin'].apply(lambda x : str(x)[0] if not pd.isnull(x) else 'NoCabin') 
# 觀察Cabin的資料後，只取出最前面的英文字母，剩下的用 NoCabin來表示


# In[55]:


data.info() # 發現Cabin已無空值


# In[56]:


data["Cabin"].unique() # 輸出有哪些客艙


# In[57]:


sns.countplot(data['Cabin'], hue=data['Survived']) # 觀察Cabin與生存率的關係


# In[58]:


# 將資料類型轉換成category，category在计算过程中会有更好的表现，接著將類別資料轉為整數。
data['Sex'] = data['Sex'].astype('category').cat.codes
data['Embarked'] = data['Embarked'].astype('category').cat.codes
data['Pclass'] = data['Pclass'].astype('category').cat.codes
data['Title1'] = data['Title1'].astype('category').cat.codes
data['Title2'] = data['Title2'].astype('category').cat.codes
data['Cabin'] = data['Cabin'].astype('category').cat.codes
data['Ticket_info'] = data['Ticket_info'].astype('category').cat.codes


# In[59]:


# 將Age值是空和非空的列分開，然後分別存到 dataAgeNull、dataAgeNotNull
dataAgeNull = data[data["Age"].isnull()]
dataAgeNotNull = data[data["Age"].notnull()]


# In[60]:


dataAgeNotNull # 輸出dataAgeNull


# In[61]:


# 輸出離群值的列
remove_outlier = dataAgeNotNull[(np.abs(dataAgeNotNull["Fare"]-dataAgeNotNull["Fare"].mean())>(4*dataAgeNotNull["Fare"].std()))|
                      (np.abs(dataAgeNotNull["Family_Size"]-dataAgeNotNull["Family_Size"].mean())>\
                       (4*dataAgeNotNull["Family_Size"].std()))]
remove_outlier


# In[62]:


len(remove_outlier) # 總共26列


# In[63]:


# 使用隨機森林來推測年齡
rfModel_age = RandomForestRegressor(n_estimators=2000,random_state=42)

# 用來補 Age值的feature
ageColumns = ['Embarked', 'Fare', 'Pclass', 'Sex', 'Family_Size', 'Title1', 'Title2','Cabin','Ticket_info']
# 拿有Age值的列來訓練模型
rfModel_age.fit(remove_outlier[ageColumns], remove_outlier["Age"])

# 訓練好的模型拿去補 Age值為空的列
ageNullValues = rfModel_age.predict(X=dataAgeNull[ageColumns])
dataAgeNull.loc[:,"Age"] = ageNullValues

# 補完值重新將剛剛拆開的兩個列合併
data = dataAgeNull.append(dataAgeNotNull)
# 更新index
data.reset_index(inplace=True, drop=True)


# In[64]:


data.info() # 補值完成，Survived有缺值是因為正解在kaggle


# In[65]:


# 用將Survived有值的跟 NaN的分開，並用PassengerId排序
dataTrain = data[pd.notnull(data['Survived'])].sort_values(by=["PassengerId"])
dataTest = data[~pd.notnull(data['Survived'])].sort_values(by=["PassengerId"])


# In[66]:


dataTrain # 輸出dataTrain


# In[67]:


dataTest # 輸出dataTest


# In[68]:


# 看一下dataTrain有哪些feature，之後要選有用的feature做
dataTrain.columns


# In[69]:


# 擷取有用的feature
dataTrain = dataTrain[['Survived', 'Age', 'Embarked', 'Fare',  'Pclass', 'Sex', 'Family_Size', 'Title2','Ticket_info','Cabin']]
dataTest = dataTest[['Age', 'Embarked', 'Fare', 'Pclass', 'Sex', 'Family_Size', 'Title2','Ticket_info','Cabin']]


# In[70]:


dataTrain


# ## Model training

# 這裡是在做gridsearchcv，但作者只放在備註，沒有用在模型內。
# ```
# # rf = RandomForestClassifier(oob_score=True, random_state=1, n_jobs=-1)
# # param_grid = { "criterion" : ["gini", "entropy"], "min_samples_leaf" : [1, 5, 10], "min_samples_split" : [2, 4, 10, 12, 16, 20], "n_estimators": [50, 100, 400, 700, 1000]}
# # gs = GridSearchCV(estimator=rf, param_grid=param_grid, scoring='accuracy', cv=3, n_jobs=-1)
# 
# # gs = gs.fit(dataTrain.iloc[:, 1:], dataTrain.iloc[:, 0])
# 
# # print(gs.best_score_)
# # print(gs.best_params_)
# ```

# In[71]:


# 跑隨機森林模型，survived當y、其他dataTrain裡的feature當x
from sklearn.ensemble import RandomForestClassifier
 
rf = RandomForestClassifier(criterion='gini', 
                             n_estimators=1000,
                             min_samples_split=12,
                             min_samples_leaf=1,
                             oob_score=True,
                             random_state=1,
                             n_jobs=-1) 

rf.fit(dataTrain.iloc[:, 1:], dataTrain.iloc[:, 0])
print("%.4f" % rf.oob_score_)
# 由于随机决策树生成过程采用的Boostrap，所以在一棵树的生成过程并不会使用所有的样本，未使用的样本就叫（out_of_bag）oob袋外样本。
# 通过袋外样本，可以评估这个树的准确度；此外，其他子树按这个原理评估。最后，取平均值即是随机森林算法的性能。


# In[72]:


pd.DataFrame(dataTrain.iloc[:, 1:].columns)


# In[73]:


# 看一下feature_importances_
pd.DataFrame(rf.feature_importances_)


# In[74]:


# 合併表格，按照importances的值從上往下排序
pd.concat((pd.DataFrame(dataTrain.iloc[:, 1:].columns, columns = ['variable']), 
           pd.DataFrame(rf.feature_importances_, columns = ['importance'])), 
          axis = 1).sort_values(by='importance', ascending = False)[:20]


# ## Submit

# In[75]:


# 預測datatest裡的乘客的生存或死亡
rf_res =  rf.predict(dataTest)


# In[76]:


# 把預測值放進submit裡面
submit['Survived'] = rf_res
# 值的資料型態轉成整數
submit['Survived'] = submit['Survived'].astype('int')
# 輸出
submit.to_csv('submit.csv', index= False)


# In[77]:


submit # 看我們預測結果，準備上傳到 kaggle


# In[ ]:




