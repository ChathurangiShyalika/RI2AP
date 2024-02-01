!pip install nixtlats>=0.1.0
import pandas as pd
from nixtlats import TimeGPT
import nltk
from nltk.tokenize import word_tokenize
import numpy as np
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.metrics import mean_squared_error
from math import sqrt
import numpy as np
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn.metrics import  classification_report,confusion_matrix

timegpt = TimeGPT(
    # defaults to os.environ.get("TIMEGPT_TOKEN")
    token = '###ADD YOUR TIMEGPT TOKEN HERE###'
)

"""#Loading df"""
data_path = "./FF_Dataset_6hour_run.csv"
df = pd.read_csv(data_path)
df.shape

df['_time'] = pd.to_datetime(df['_time'])
df['_time'] = df['_time'].dt.tz_localize(None)
pm_df=df


#change column names accordingly
#for i in all column names except desc
column_names=['SJointAngle_R03','Description']

df=df[column_names]
nltk.download('punkt')
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

def calculate_average(row):
    numeric_values = [value for value in row if value is not None]
    return sum(numeric_values) / len(numeric_values) if len(numeric_values) > 0 else 0

df['Average'] = df['Tokenized_Description1'].apply(calculate_average)
features_df=df[[column_names[0],'Average']]
features_df=features_df.rename(columns={"Average": "Description"})
features_df

pm_df['Description']=features_df['Description']

"""#splitting data"""
train = pm_df[:int(0.8*(len(pm_df)))]
valid = pm_df[int(0.8*(len(pm_df))):]

"""##creating training data"""
train_2=train[column_names].round(0)
train_2

"""##Unique value count"""
unique, counts = np.unique(train_2.Description.round(0), return_counts=True)
dict(zip(unique, counts))

train_2.tail(20)

"""###FINAL TRAINING DATA

train_new.
This is the training data that we have used to train our model

"""

train_new=train_2.iloc[:95710]
unique, counts = np.unique(train_new.Description.round(0), return_counts=True)
dict(zip(unique, counts))
train_new.tail(20)

"""#SCALING DATA"""

scalery = StandardScaler()

# Fit and transform the data (for 'prediction')
df_scaled = scalery.fit_transform(train_new.values)
train_scaled = pd.DataFrame(df_scaled, index=train_new.index, columns=train_new.columns)
train_scaled

time=train['_time'].iloc[:95710]
train_new_scaled=pd.concat([time,train_scaled],axis=1)
train_new_scaled

"""#Creating new df which contains our final training data."""

new=train_new_scaled
new=new.melt(id_vars=["_time"],
        var_name="unique_id",
        value_name="Value")

new.rename(columns={'_time': 'ds' },inplace=True, errors='raise')
train_new_scaled.tail(20)

"""#Predicting"""
timegpt_fcst_pred_both_df = timegpt.forecast(model='timegpt-1-long-horizon',finetune_steps=90,
    df=new, h=2000,freq='100ms', level=[90],
    time_col='ds', target_col='Value',
)
#level 80,90 gives same value. Max h can be 2000 only
#fintune_steps is same as epochs. It means that the model is iterating or getting trained 90 times.

timegpt_fcst_pred_both_df

predicted_df = timegpt_fcst_pred_both_df.pivot(index="ds", columns="unique_id", values="TimeGPT")
predicted_df

predicted_df = predicted_df.reset_index()
predicted_df

prediction=predicted_df['Description'].astype('float').round(0)
prediction

unique, counts = np.unique(prediction, return_counts=True)
dict(zip(unique, counts))

x=prediction.shape[0]

"""#Redefining the predicted description values

"""

for j in range(0,x):
    if prediction[j] == -1:
      prediction[j]=0
    # elif prediction[j] == -0:
    #   prediction[j]=4
    # elif prediction[j] == 2:
    #   prediction[j]=5
    # elif prediction[j] == 9:
    #   prediction[j]=6
    # elif prediction[j] == 3:
    #   prediction[j]=8
    elif prediction[j] == -0:
      prediction[j]= 9
prediction
#{0.0: 19406, 4.0: 9043, 5.0: 4405, 6.0: 5904, 8.0: 3306, 9.0: 125}

unique, counts = np.unique(train_new['Description'].round(0), return_counts=True)
dict(zip(unique, counts))

unique, counts = np.unique(prediction, return_counts=True)
dict(zip(unique, counts))

"""#Creating testing data from training data
"""

testing=train_2.iloc[95710:97710].round(0)
testing

testing_desc=testing['Description']
testing_desc

"""#Scaling testing data"""
scaler = StandardScaler()

# Fit and transform the data (for 'prediction')
df_scaled = scaler.fit_transform(testing.values)
test_scaled = pd.DataFrame(df_scaled, index=testing.index, columns=testing.columns)
test_scaled

y_test=test_scaled['Description']
y_test

"""#Calculating RMSE,MSE for Description"""

# Calculate RMSE
mse = mean_squared_error(y_test, prediction)
rmse = np.sqrt(mse)

print(f"RMSE {column_names[1]}:{rmse:.1f}")
print(f"MSE {column_names[1]}: {mse:.1f}")
print(f"RMSE {column_names[1]}:{rmse:.4f}")
print(f"MSE {column_names[1]}: {mse:.4f}")

predicted_df[f'{column_names[0]}']

prediction_2 = predicted_df[f'{column_names[0]}']
y_test_2 = test_scaled[f'{column_names[0]}']

# Calculate RMSE
mse = mean_squared_error(y_test_2, prediction_2)
rmse = np.sqrt(mse)

print(f"RMSE {column_names[0]}:{rmse:.1f}")
print(f"MSE {column_names[0]}: {mse:.1f}")
print(f"RMSE {column_names[0]}:{rmse:.4f}")
print(f"MSE {column_names[0]}: {mse:.4f}")

"""#Total rmse mse"""

all_label=np.concatenate((y_test,y_test_2))
all_pred=np.concatenate((prediction,prediction_2))
mse_all=mean_squared_error(all_label, all_pred)
rmse_all=np.sqrt(mse_all)
print(f"All (RMSE): {rmse_all:.1f}")
print(f"All (MSE): {mse_all:.1f}")
print(f"All (RMSE): {rmse_all:.4f}")
print(f"All (MSE): {mse_all:.4f}")

"""#CLASSIFICATION REPORT"""

y_test

# target_names = ['0.0','4.0','5.0','8.0', '9.0']
target_names = ['0.0', '9.0']
print(classification_report(y_test, prediction, target_names=target_names))
confusion_matrix(y_test, prediction)

"""#DESCALING

##Desacaling feature 1
"""

prediction_2

"""**descaling using scalery (used for training data)**"""

prediction_2 = prediction_2.values.reshape(-1, 1)

# Concatenate with zeros
descaled_pred2 = scalery.inverse_transform(np.concatenate([prediction_2, np.zeros_like(prediction_2)], axis=1))[:, 0]

descaled_pred2.round(6)

"""**descaling using scaler (used for testing data)**"""

# prediction_2 = prediction_2.reshape(-1, 1)

# Concatenate with zeros
descaled_pred_2 = scaler.inverse_transform(np.concatenate([prediction_2, np.zeros_like(prediction_2)], axis=1))[:, 0]

descaled_pred_2.round(6)

testing

"""#Creating final df"""

testing

y=testing['Description'].index

test_desc=pd.DataFrame(testing_desc,columns =['Description'])
test_desc

test_desc.to_csv(("Testing_DataFrame_TimeGPT.csv"))

ind=pd.DataFrame(y,columns =['Original index'])
temp=pd.DataFrame(prediction,columns =['Description']) #Scaled Description value

predictions_descaled_2=pd.DataFrame(descaled_pred2.round(6),columns =[f'{column_names[0]}']) #Descaled Feature 1 Predicted value
predictions_descaled_2=pd.concat([ind,predictions_descaled_2], axis=1)
merged_df = pd.merge(predictions_descaled_2, temp, left_index=True, right_index=True)

merged_df.to_csv((f"{column_names[0]}_TimeGPT.csv"))

lc=pd.read_csv(f"{column_names[0]}_TimeGPT.csv")
lc

t=train[['Description',f"{column_names[0]}"]].iloc[95710:]
t

unique, counts = np.unique(t.Description, return_counts=True)

dict(zip(unique, counts))

t_df = t.reset_index()
t_df

t_df[t_df['Description']==9].index



