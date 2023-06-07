#!/usr/bin/env python
# coding: utf-8

# In[1]:


##PYTHON PROJECT ON SPOTIFY DATASET BY SIDDHESH##
    
#Import the Python Libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[114]:


#Reading the first 5 rows of dataset

df_tracks = pd.read_csv('/Users/siddheshkadam/Downloads/archive/tracks.csv')
df_tracks


# In[15]:


#Checking if there are any null values in the dataset

pd.isnull(df_tracks).sum()


# In[13]:


#Getting the basic information of the dataset

df_tracks.info()


# In[17]:


#Getting the 10 most least popular songs from the set

df_tracks.sort_values('popularity', ascending = True).head(10)


# In[24]:


#Getting the basic statistics of the set

df_tracks.describe().transpose()


# In[39]:


#Getting the 10 most popular songs in the set

most_popular = df_tracks.query('popularity>90', inplace = False).sort_values('popularity', ascending = False).head(10)
most_popular


# In[40]:


#set index as the 'release date' column and get the first 5 rows

df_tracks.set_index('release_date', inplace = True)
df_tracks.index=pd.to_datetime(df_tracks.index)
df_tracks.head()


# In[43]:


#Get the artist at the 18th row

df_tracks[['artists']].iloc[18]


# In[58]:


#Changed the miliseconds to seconds for duration

df_tracks.duration.head()


# In[63]:


#Checking the Correlation HeatMap Between Variables using Seaborn library

corr_df = df_tracks.drop(['key', 'mode', 'explicit'], axis=1).corr(method='pearson', numeric_only=True)
plt.figure(figsize=(14, 6))
heatmap = sns.heatmap(corr_df, annot=True, fmt='.1g', vmin=-1, vmax=1, center=0, cmap='inferno', linewidths=1, linecolor='Black')
heatmap.set_title('Correlation HeatMap Between Variables')
heatmap.set_xticklabels(heatmap.get_xticklabels(), rotation=90)


# In[65]:


#Get the sample of the dataset

sample_df = df_tracks.sample(int(0.004*len(df_tracks)))


# In[66]:


print(len(sample_df))


# In[73]:


#After the previous plot we can see there is a positive correlation between loudness and energy coloumn
#Hence, we are plotting the regression line between them

plt.figure(figsize=(10,6))
sns.regplot(data = sample_df, y="loudness", x="energy", color = "c").set(title = "Loudness vs Energy Correlation")

#We can see there is a positive Correlation between Loudness and Energy


# In[75]:


#After the previous plot we can see there is a also a correlation between popularity and acousticness coloumn
#Hence, we are plotting the regression line between them

plt.figure(figsize=(10,6))
sns.regplot(data = sample_df, y="popularity", x="acousticness", color = "b").set(title = "Popularity vs Acousticness Correlation")


# In[100]:


##Getting the second dataset consisting of Genres and do some more analysis


df_genre = pd.read_csv('/Users/siddheshkadam/Downloads/SpotifyFeatures.csv')


# In[113]:


df_genre


# In[105]:


#Plotting bar graph for Genres and Duration in milliseconds

plt.title("Duration of the songs in the different genres")
sns.color_palette("rocket", as_cmap=True)
sns.barplot(y='genre', x='duration_ms', data=df_genre)
plt.xlabel("Duration in milliseconds")
plt.ylabel("Genres")


# In[112]:


#Getting the top 5 Genres by most Popularity in the dataset

sns.set_style(style = "darkgrid")
plt.figure(figsize=(10,5))
famous = df_genre.sort_values("popularity", ascending=False).head(10)
sns.barplot(y='genre', x='popularity', data=famous).set(title="Top 5 Genres by Popularity")
plt.xlabel("Popularity")
plt.ylabel("Genres")


# In[ ]:


#THE END

