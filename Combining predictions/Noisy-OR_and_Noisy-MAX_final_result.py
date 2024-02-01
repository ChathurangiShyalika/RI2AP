import pandas as pd
from sklearn.metrics import mean_squared_error,classification_report
import numpy as np
import numpy as np

#Importing predictions for all the features
#Below codes relate to RI2AP model for merging all the eight anomaly types and normal type.
#Code is for merging noisyor results. Change path accordingly for noisymax results.
#For other baselines, change the paths for the saved predictions from the models.
df1 = pd.read_csv("../R03 crashed tail_noisyor.csv")
df2 = pd.read_csv("../R04 crashed nose_noisyor.csv")
df3 = pd.read_csv("../body2 removed_noisyor.csv")
df4 = pd.read_csv("../bothbodies and nose removed_noisyor.csv")
df5 = pd.read_csv("../door2_timeout_noisyor.csv")
df6 = pd.read_csv("../estopped_noisyor.csv")
df7 = pd.read_csv("../none_noisyor.csv")
df8 = pd.read_csv("../topbody and nose removed_noisyor.csv")
df9 = pd.read_csv("../Nosecone Removed_linear_noisyor.csv")

#merging files
frames = [df1, df2, df3,df4,df5,df6,df7,df8, df9]
merged_df = pd.concat(frames)

#Calculate the
rmse_var1 = np.sqrt(mean_squared_error(merged_df['y'], merged_df['Description']))
rmse_var1

#get different prediction counts
merged_df['y'].value_counts()

temp_desc=merged_df['y'].round(2)
x=temp_desc.size

temp_desc=temp_desc.values
temp_desc

for j in range(0,x):
    if temp_desc[j]< 1.5:
      temp_desc[j]=13
    elif temp_desc[j]>= 1.5 and temp_desc[j]<4.815:
      temp_desc[j]=14
    elif temp_desc[j]>=4.815 and temp_desc[j]< 5.15:
      temp_desc[j]=15
    elif temp_desc[j]>=5.15 and temp_desc[j]<5.74:
      temp_desc[j]=16
    elif temp_desc[j]>=5.74 and temp_desc[j]<6.6:
      temp_desc[j]=17
    elif temp_desc[j]>=6.6 and temp_desc[j]<7.735:
      temp_desc[j]=18
    elif temp_desc[j]>=7.735 and temp_desc[j]<8.4:
      temp_desc[j]=19
    elif temp_desc[j]>=8.4 and temp_desc[j]<10:
      temp_desc[j]=20
    elif temp_desc[j]>= 10:
      temp_desc[j]=21

#get unique counts for each predicted class
unique, counts = np.unique(temp_desc, return_counts=True)
dict(zip(unique, counts))

d=pd.DataFrame(temp_desc, columns=['newclass'])
d['new_index'] = range(1, len(d) + 1)

temp_desc1=merged_df['Description'].round(2)
temp_desc1=temp_desc1.values

for j in range(0,x):
    if temp_desc1[j]==0.:
      temp_desc1[j]=13
    elif temp_desc1[j]==4.67:
      temp_desc1[j]=14
    elif temp_desc1[j]==4.5:
      temp_desc1[j]=15
    elif temp_desc1[j]==5.5:
      temp_desc1[j]=16
    elif temp_desc1[j]==6.25:
      temp_desc1[j]=17
    elif temp_desc1[j]==7.67:
      temp_desc1[j]=18
    elif temp_desc1[j]==8.:
      temp_desc1[j]=19
    elif temp_desc1[j]==9.:
      temp_desc1[j]=20
    elif temp_desc1[j]>= 12.:
      temp_desc1[j]=21

#get unique counts for each predicted class
unique, counts = np.unique(temp_desc1, return_counts=True)
dict(zip(unique, counts))

d1=pd.DataFrame(temp_desc1, columns=['original'])
d1['new_index'] = range(1, len(d) + 1)


merged_df = pd.merge(d, d1, on='new_index', how='inner')
merged_df

#get final predicted classes
classification_report_var1 = classification_report(merged_df['newclass'], merged_df['original'])
print(classification_report_var1)