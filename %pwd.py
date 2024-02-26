#!/usr/bin/env python
# coding: utf-8

# # Mall clustering project
# 

# In[2]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')


# In[3]:


df = pd.read_csv("/Users/sahilamirza/Documents/Datasets/Mall_Customers.csv")


# In[4]:


df.head()


# # Univariate Analysis

# In[5]:


df.describe()


# In[6]:


sns.distplot(df['Annual Income (k$)']);


# In[7]:


df.columns


# In[8]:


columns = ['Age', 'Annual Income (k$)','Spending Score (1-100)']
for i in columns:
    plt.figure()
    sns.distplot(df[i])


# In[9]:


melted_df = pd.melt(df, id_vars=['Gender'], value_vars=['Annual Income (k$)'])

sns.displot(data=melted_df, x='value', hue='Gender', multiple='stack', kind='kde', fill=True)


# In[33]:


print(df['Gender'].dtype)


# In[34]:


df['Gender'] = df['Gender'].astype('category')


# In[35]:


print(melted_df.head())


# In[10]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


df_long = df[['Age', 'Annual Income (k$)', 'Spending Score (1-100)', 'Gender']].copy()
df_long = pd.melt(df_long, id_vars='Gender', var_name='Variable', value_name='Value')

for var in df_long['Variable'].unique():
    plt.figure()
    sns.kdeplot(x='Value', hue='Gender', data=df_long, color='skyblue', edgecolor='royalblue', shade=True)
    plt.xlabel(var)
    plt.ylabel('Density')
    plt.title(f'Distribution of {var} by Gender')
    plt.show()


# In[11]:


columns = ['Age', 'Annual Income (k$)','Spending Score (1-100)']
for i in columns:
    plt.figure()
    sns.boxplot(data=df,x='Gender',y=df[i])


# In[12]:


df['Gender'].value_counts(normalize=True)


# # Bivariate Analysis
# 

# In[13]:


sns.scatterplot(data=df, x='Annual Income (k$)',y='Spending Score (1-100)' )


# In[14]:


sns.pairplot(df,hue='Gender')


# In[15]:


df.groupby('Gender')[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']].mean()


# In[31]:


numeric_df = df.select_dtypes(include=['number'])
correlation_matrix = numeric_df.corr()


# In[33]:


numeric_df.corr()


# In[34]:


sns.heatmap(numeric_df.corr(),annot=True,cmap='coolwarm')



# # Clustering - Univariate, Bivariate, Multivariate

# In[35]:


clustering1 = KMeans(n_clusters=3)


# In[36]:


clustering1.fit(df[['Annual Income (k$)']])


# In[37]:


clustering1.labels_


# In[38]:


df['Income Cluster'] = clustering1.labels_
df.head()


# In[39]:


df['Income Cluster'].value_counts()


# In[40]:


clustering1.inertia_


# In[41]:


intertia_scores=[]
for i in range(1,11):
    kmeans=KMeans(n_clusters=i)
    kmeans.fit(df[['Annual Income (k$)']])
    intertia_scores.append(kmeans.inertia_)


# In[42]:


intertia_scores


# In[43]:


plt.plot(range(1,11),intertia_scores)


# In[44]:


df.columns


# In[46]:


df.groupby('Income Cluster')[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']].mean()


# In[47]:


#Bivariate Clustering


# In[48]:


clustering2 = KMeans(n_clusters=5)
clustering2.fit(df[['Annual Income (k$)','Spending Score (1-100)']])
df['Spending and Income Cluster'] =clustering2.labels_
df.head()


# In[49]:


intertia_scores2=[]
for i in range(1,11):
    kmeans2=KMeans(n_clusters=i)
    kmeans2.fit(df[['Annual Income (k$)','Spending Score (1-100)']])
    intertia_scores2.append(kmeans2.inertia_)
plt.plot(range(1,11),intertia_scores2)


# In[50]:


centers =pd.DataFrame(clustering2.cluster_centers_)
centers.columns = ['x','y']


# In[51]:


plt.figure(figsize=(10,8))
plt.scatter(x=centers['x'],y=centers['y'],s=100,c='black',marker='*')
sns.scatterplot(data=df, x ='Annual Income (k$)',y='Spending Score (1-100)',hue='Spending and Income Cluster',palette='tab10')
plt.savefig('clustering_bivaraiate.png')


# In[52]:


pd.crosstab(df['Spending and Income Cluster'],df['Gender'],normalize='index')


# In[66]:


df.groupby('Spending and Income Cluster')[['Age', 'Annual Income (k$)',
       'Spending Score (1-100)']].mean()


# In[55]:


#mulivariate clustering 
from sklearn.preprocessing import StandardScaler


# In[56]:


scale = StandardScaler()


# In[57]:


df.head()


# In[58]:


dff = pd.get_dummies(df,drop_first=True)
dff.head()


# In[59]:


dff.columns


# In[60]:


dff = dff[['Age', 'Annual Income (k$)', 'Spending Score (1-100)','Gender_Male']]
dff.head()


# In[61]:


dff = scale.fit_transform(dff)


# In[62]:


dff = pd.DataFrame(scale.fit_transform(dff))
dff.head()


# In[63]:


intertia_scores3=[]
for i in range(1,11):
    kmeans3=KMeans(n_clusters=i)
    kmeans3.fit(dff)
    intertia_scores3.append(kmeans3.inertia_)
plt.plot(range(1,11),intertia_scores3)


# In[64]:


df


# In[65]:


df.to_csv('Clustering.csv')


# In[ ]:




