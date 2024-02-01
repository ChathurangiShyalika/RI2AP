import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from sklearn.metrics import mean_squared_error,classification_report
import numpy as np
import numpy as np

#Importing predictions for all the features
#Below codes relate to RI2AP model for one anomaly type(Nosecone Removed) only. Change anomaly type accordingly to get final predicted dataset for RI2AP model.
#For other baselines, change the paths for the saved predictions from the models.
#Change the path for saved prediction files accordingly.
df1 = pd.read_csv("/content/drive/MyDrive/ICML 2024/Linear model/5_nosecone removed_4/LJointAngle_R01_Nosecone Removed_linear.csv")
df1=df1.rename(columns={"Unnamed: 0": "indexa","Variable1":"x1","Variable2":"y1"})
df1=df1[['indexa','x1','y1']]

df2 = pd.read_csv("/content/drive/MyDrive/ICML 2024/Linear model/5_nosecone removed_4/LoadCell_R02_Nosecone Removed_linear.csv")
df2=df2.rename(columns={"Unnamed: 0": "indexa","Variable1":"x2","Variable2":"y2"})
df2=df2[['indexa','x2','y2']]

df3 = pd.read_csv("/content/drive/MyDrive/ICML 2024/Linear model/5_nosecone removed_4/LoadCell_R03_Nosecone Removed_linear.csv")
df3=df3.rename(columns={"Unnamed: 0": "indexa","Variable1":"x3","Variable2":"y3"})
df3=df3[['indexa','x3','y3']]

df4 = pd.read_csv("/content/drive/MyDrive/ICML 2024/Linear model/5_nosecone removed_4/SJointAngle_R03_Nosecone Removed_linear.csv")
df4=df4.rename(columns={"Unnamed: 0": "indexa","Variable1":"x4","Variable2":"y4"})
df4=df4[['indexa','x4','y4']]

df5 = pd.read_csv("/content/drive/MyDrive/ICML 2024/Linear model/5_nosecone removed_4/VFD2_Nosecone Removed_linear.csv")
df5=df5.rename(columns={"Unnamed: 0": "indexa","Variable1":"x5","Variable2":"y5"})
df5=df5[['indexa','x5','y5']]

df6 = pd.read_csv("/content/drive/MyDrive/ICML 2024/Linear model/5_nosecone removed_4/BJointAngle_R03_Nosecone Removed_linear.csv")
df6=df6.rename(columns={"Unnamed: 0": "indexa","Variable1":"x6","Variable2":"y6"})
df6=df6[['indexa','x6','y6']]

df7 = pd.read_csv("/content/drive/MyDrive/ICML 2024/Linear model/5_nosecone removed_4/BJointAngle_R04_Nosecone Removed_linear.csv")
df7=df7.rename(columns={"Unnamed: 0": "indexa","Variable1":"x7","Variable2":"y7"})
df7=df7[['indexa','x7','y7']]

df8 = pd.read_csv("/content/drive/MyDrive/ICML 2024/Linear model/5_nosecone removed_4/Potentiometer_R01_Nosecone Removed_linear.csv")
df8=df8.rename(columns={"Unnamed: 0": "indexa","Variable1":"x8","Variable2":"y8"})
df8=df8[['indexa','x8','y8']]

df9 = pd.read_csv("/content/drive/MyDrive/ICML 2024/Linear model/5_nosecone removed_4/LJointAngle_R04_Nosecone Removed_linear.csv")
df9=df9.rename(columns={"Unnamed: 0": "indexa","Variable1":"x9","Variable2":"y9"})
df9=df9[['indexa','x9','y9']]

df10 = pd.read_csv("/content/drive/MyDrive/ICML 2024/Linear model/5_nosecone removed_4/LoadCell_R01_Nosecone Removed_linear.csv")
df10=df10.rename(columns={"Unnamed: 0": "indexa","Variable1":"x10","Variable2":"y10"})
df10=df10[['indexa','x10','y10']]

df11 = pd.read_csv("/content/drive/MyDrive/ICML 2024/Linear model/5_nosecone removed_4/Potentiometer_R02_Nosecone Removed_linear.csv")
df11=df11.rename(columns={"Unnamed: 0": "indexa","Variable1":"x11","Variable2":"y11"})
df11=df11[['indexa','x11','y11']]

df12 = pd.read_csv("/content/drive/MyDrive/ICML 2024/Linear model/5_nosecone removed_4/LoadCell_R04_Nosecone Removed_linear.csv")
df12=df12.rename(columns={"Unnamed: 0": "indexa","Variable1":"x12","Variable2":"y12"})
df12=df12[['indexa','x12','y12']]

df13 = pd.read_csv("/content/drive/MyDrive/ICML 2024/Linear model/5_nosecone removed_4/Potentiometer_R03_Nosecone Removed_linear.csv")
df13=df13.rename(columns={"Unnamed: 0": "indexa","Variable1":"x13","Variable2":"y13"})
df13=df13[['indexa','x13','y13']]

df14 = pd.read_csv("/content/drive/MyDrive/ICML 2024/Linear model/5_nosecone removed_4/Potentiometer_R04_Nosecone Removed_linear.csv")
df14=df14.rename(columns={"Unnamed: 0": "indexa","Variable1":"x14","Variable2":"y14"})
df14=df14[['indexa','x14','y14']]

df15 = pd.read_csv("/content/drive/MyDrive/ICML 2024/Linear model/5_nosecone removed_4/RJointAngle_R04_Nosecone Removed_linear.csv")
df15=df15.rename(columns={"Unnamed: 0": "indexa","Variable1":"x15","Variable2":"y15"})
df15=df15[['indexa','x15','y15']]

df16 = pd.read_csv("/content/drive/MyDrive/ICML 2024/Linear model/5_nosecone removed_4/SJointAngle_R02_Nosecone Removed_linear.csv")
df16=df16.rename(columns={"Unnamed: 0": "indexa","Variable1":"x16","Variable2":"y16"})
df16=df16[['indexa','x16','y16']]

df17 = pd.read_csv("/content/drive/MyDrive/ICML 2024/Linear model/5_nosecone removed_4/SJointAngle_R04_Nosecone Removed_linear.csv")
df17=df17.rename(columns={"Unnamed: 0": "indexa","Variable1":"x17","Variable2":"y17"})
df17=df17[['indexa','x17','y17']]

df18 = pd.read_csv("/content/drive/MyDrive/ICML 2024/Linear model/5_nosecone removed_4/TJointAngle_R04_Nosecone Removed_linear.csv")
df18=df18.rename(columns={"Unnamed: 0": "indexa","Variable1":"x18","Variable2":"y18"})
df18=df18[['indexa','x18','y18']]

df19 = pd.read_csv("/content/drive/MyDrive/ICML 2024/Linear model/5_nosecone removed_4/UJointAngle_R03_Nosecone Removed_linear.csv")
df19=df19.rename(columns={"Unnamed: 0": "indexa","Variable1":"x19","Variable2":"y19"})
df19=df19[['indexa','x19','y19']]

df20 = pd.read_csv("/content/drive/MyDrive/ICML 2024/Linear model/5_nosecone removed_4/VFD1_Nosecone Removed_linear.csv")
df20=df20.rename(columns={"Unnamed: 0": "indexa","Variable1":"x20","Variable2":"y20"})
df20=df20[['indexa','x20','y20']]

#Merging individual prediction files
merged_df = pd.merge(df1, df2, on='indexa', how='outer')
merged_df = pd.merge(merged_df, df3, on='indexa', how='outer')
merged_df = pd.merge(merged_df, df4, on='indexa', how='outer')
merged_df = pd.merge(merged_df, df5, on='indexa', how='outer')
merged_df = pd.merge(merged_df, df6, on='indexa', how='outer')
merged_df = pd.merge(merged_df, df7, on='indexa', how='outer')
merged_df = pd.merge(merged_df, df8, on='indexa', how='outer')
merged_df = pd.merge(merged_df, df9, on='indexa', how='outer')
merged_df = pd.merge(merged_df, df10, on='indexa', how='outer')
merged_df = pd.merge(merged_df, df11, on='indexa', how='outer')
merged_df = pd.merge(merged_df, df12, on='indexa', how='outer')
merged_df = pd.merge(merged_df, df13, on='indexa', how='outer')
merged_df = pd.merge(merged_df, df14, on='indexa', how='outer')
merged_df = pd.merge(merged_df, df15, on='indexa', how='outer')
merged_df = pd.merge(merged_df, df16, on='indexa', how='outer')
merged_df = pd.merge(merged_df, df17, on='indexa', how='outer')
merged_df = pd.merge(merged_df, df18, on='indexa', how='outer')
merged_df = pd.merge(merged_df, df19, on='indexa', how='outer')
merged_df = pd.merge(merged_df, df20, on='indexa', how='outer')
merged_df
final_df = pd.DataFrame(columns=['indexa', 'x1', 'x2','x3','x4','x5','x6','x7','x8','x9', 'x10',
                                 'x11','x12','x13', 'x14','x15','x16', 'x17','x18','x19', 'x20','y'])


for i in range(1, 21):
    temp_df = pd.DataFrame()
    temp_df['time'] = merged_df['indexa']
    temp_df['x1'] = 0
    temp_df['x2'] = 0
    temp_df['x3'] = 0
    temp_df['x4'] = 0
    temp_df['x5'] = 0
    temp_df['x6'] = 0
    temp_df['x7'] = 0
    temp_df['x8'] = 0
    temp_df['x9'] = 0
    temp_df['x10'] = 0
    temp_df['x11'] = 0
    temp_df['x12'] = 0
    temp_df['x13'] = 0
    temp_df['x14'] = 0
    temp_df['x15'] = 0
    temp_df['x16'] = 0
    temp_df['x17'] = 0
    temp_df['x18'] = 0
    temp_df['x19'] = 0
    temp_df['x20'] = 0
    temp_df['y'] = 0

    temp_df['x' + str(i)] = merged_df['x' + str(i)]
    temp_df['y'] = merged_df['y' + str(i)]

    final_df = pd.concat([final_df, temp_df], ignore_index=True)

final_df = final_df.fillna(0).astype({'x1': float, 'x2': float, 'x3': float,'x4': float,'x5': float, 'x6': float,
                                      'x7': float, 'x8': float, 'x9': float,'x10': float,
                                      'x11': float, 'x12': float,'x13': float, 'x14': float, 'x15': float,
                                      'x16': float, 'x17': float, 'x18': float,'x19': float, 'x20': float})


final_df = final_df.sort_values(by='time')
final_df


#Getting all possible predictions at each time step
final_df['new_index'] = range(1, len(final_df) + 1)
final_df


#ground truth y value
df = pd.read_csv("../FF_Dataset_6hour_run.csv")
df=df.loc[df['Description'] == "Nosecone Removed"]

#change column names accordingly

#for i in all column names except desc
column_names=['VFD2','Description']

nltk.download('punkt')  # Download the NLTK tokenizer data (if not downloaded)
df['Tokenized_Description'] = df['Description'].apply(lambda x: word_tokenize(str(x)) if pd.notnull(x) else [])
result = df[[column_names[0], 'Tokenized_Description']].values.tolist()

all_tokens = set()
_ = df['Tokenized_Description'].apply(lambda x: all_tokens.update(x) if x else all_tokens.add(None))

# List of all different tokens
different_tokens = list(all_tokens)

tokens = {'None':0.0,'Nose':1.0, 'nose': 2.0, 'Removed': 3.0, 'crashed': 4.0, 'R03': 5.0, 'Nosecone': 6.0, 'BothBodies': 7.0, 'R04': 8.0, 'Door2_TimedOut': 9.0, 'TopBody': 10.0, 'and': 11.0, 'ESTOPPED': 12.0, 'Body2': 13.0, 'tail': 14.0}

def replace_with_numeric(tokens_dict, tokens_list):
    return [tokens_dict[token] if token in tokens_dict else None for token in tokens_list]

df['Tokenized_Description1'] = df['Tokenized_Description'].apply(lambda x: replace_with_numeric(tokens, x))
df=df[['_time','Tokenized_Description','Tokenized_Description1']]

def replace_empty_with_none(tokens_list):
    return 'None' if len(tokens_list) == 0 else tokens_list

df['Tokenized_Description'] = df['Tokenized_Description'].apply(lambda x: replace_empty_with_none(x))

def calculate_average(row):
    numeric_values = [value for value in row if value is not None]
    return sum(numeric_values) / len(numeric_values) if len(numeric_values) > 0 else 0

df['Average'] = df['Tokenized_Description1'].apply(calculate_average)
features_df=df[['Average']]
features_df=features_df.rename(columns={"Average": "Description"})
features_df=features_df.reset_index()

#Token value for this anomaly type is 4.5. Using it below
df_max_prob['Description'] = 4.5
df_max_prob
final_df=df_max_prob
final_df


#calculating RMSE values
rmse_var1 = np.sqrt(mean_squared_error(final_df['y'], final_df['Description']))
rmse_var1

classification_report_var1 = classification_report(final_df['y'].round(0), final_df['Description'].round(0))
print(classification_report_var1)


final_df.to_csv("Nosecone Removed_linear_noisyor.csv")


