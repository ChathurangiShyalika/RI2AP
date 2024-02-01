!pip install --quiet pytorch-lightning
!pip install --quiet tqdm

import nltk
from nltk.tokenize import word_tokenize
import pandas as pd

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from torch.optim.lr_scheduler import ReduceLROnPlateau

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
import pickle


df = pd.read_csv(“FF_Dataset_6hour_run.csv”)

df=df.loc[df['Description'] == "BothBodies and Nose Removed"]
df


import nltk
from nltk.tokenize import word_tokenize


#change column names accordingly

#for i in all column names except desc
column_names=['SJointAngle_R03','Description']

df=df[column_names]
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

def replace_empty_with_none(tokens_list):
    return 'None' if len(tokens_list) == 0 else tokens_list

df['Tokenized_Description'] = df['Tokenized_Description'].apply(lambda x: replace_empty_with_none(x))

def calculate_average(row):
    numeric_values = [value for value in row if value is not None]
    return sum(numeric_values) / len(numeric_values) if len(numeric_values) > 0 else 0

df['Average'] = df['Tokenized_Description1'].apply(calculate_average)

split_index = int(0.8 * len(df))

# Split the data into training and validation sets
df_train = df.iloc[:split_index]
df_test = df.iloc[split_index:]

features_df=df[[column_names[0],'Average']]
features_df=features_df.rename(columns={"Average": "Description"})

p_y=features_df.values.tolist()

result = df[[column_names[0],'Tokenized_Description']]
result

def replace_empty_with_none(tokens_list):
    return 'None' if len(tokens_list) == 0 else tokens_list


result['Tokenized_Description'] = result['Tokenized_Description'].apply(lambda x: replace_empty_with_none(x))

x=result.values.tolist()

m = len(x)
p_x = [x[:i+1] for i in range(m-1)]

#USING LINEAR LAYERS

import torch
import torch.nn as nn
import torch.nn.functional as F

class twoD_predict(nn.Module):

  def __init__(self):

    super().__init__()
    self.f1 = nn.Linear(3,1000)
    self.f2 = nn.Linear(1000,1000)
    self.f3 = nn.Linear(3,1000)
    self.f4 = nn.Linear(1000,1000)
    self.c1 = nn.Linear(1000,1)
    self.c2 = nn.Linear(1000,1)

  def aggr1(self,n_obvs):

    var = torch.div(torch.sum(torch.pow(n_obvs-torch.mean(n_obvs),2)),len(n_obvs))
    eps = 1e03
    centralized_n_obvs = torch.div(n_obvs-torch.mean(n_obvs),var-1+eps)
    order_means = [torch.mean(torch.pow(centralized_n_obvs,order)) for order in range(3)]
    return torch.tensor(order_means)

  def aggr2(self,c_obvs):

    n_obvs = c_obvs
    var = torch.div(torch.sum(torch.pow(n_obvs-torch.mean(n_obvs),2)),len(n_obvs))
    eps = 1e03
    centralized_n_obvs = torch.div(n_obvs-torch.mean(n_obvs),var-1+eps)
    order_means = [torch.mean(torch.pow(centralized_n_obvs,order)) for order in range(3)]
    return torch.tensor(order_means)

  def forward(self,x):

    n_obvs = torch.tensor([float(i[0]) for i in x])  # Convert to float explicitly
    c_obvs = torch.tensor([tokens[j] for i in x for j in i[1]], dtype=torch.float) # Assuming tokens are float values

    n_obvs = self.aggr1(n_obvs)
    l_n = F.leaky_relu(self.f1(n_obvs))
    l_n = F.leaky_relu(self.f2(l_n))
    c_n = F.leaky_relu(self.c1(l_n))

    c_obvs = self.aggr2(c_obvs)
    l_c = F.leaky_relu(self.f3(c_obvs))
    l_c = F.leaky_relu(self.f4(l_c))
    c_c = F.leaky_relu(self.c2(l_c))

    return [c_n,c_c]

  def train(self,
            epochs = 100):

    optimizer = torch.optim.AdamW(self.parameters())
    n = len(p_x)
    for i in range(epochs):
      predictions = [self(p_x[j]) for j in range(n)]
      gts = torch.tensor(p_y)
      loss = 0.0
      for j in range(n):
        for l in range(2):
          loss += torch.pow(predictions[j][l]-gts[j][l],2)
      loss /= n
      loss.backward()
      print (loss.item())
      optimizer.step()
      optimizer.zero_grad()

obj1 = twoD_predict()

obj1.train()


filename = (f"{column_names[0]}_linearmodel.sav")
pickle.dump(obj1, open(filename, 'wb'))

x=df_test.values.tolist()

df_test1=df_test[[column_names[0],'Tokenized_Description']]


# Specify the start_index
start_index = df_test1.index[0]

# Initialize a list to store the results
combined_predictions = []

# Loop through different end_index values
for i in range(100):  # Adjust the number of iterations as needed, taking 100 for now
    end_index = start_index + 2 * i #2-gap between rows

    df_test11 = df_test1.loc[start_index:end_index].values.tolist()
    test_predictions = obj1(df_test11)
    combined_predictions.append((start_index, end_index, test_predictions))

# Create a list to store all tensor values
all_tensor_values_list = []

# Process each set of predictions
for _, end_index, predictions in combined_predictions:
    tensor_values_list = [float(tensor_val.item()) for tensor_val in predictions]
    all_tensor_values_list.append(tensor_values_list)

columns = ['Variable1', 'Variable2']
df_result = pd.DataFrame(all_tensor_values_list, columns=columns)

df_result['index'] = [end_index + 1 for _, end_index, _ in combined_predictions]

df_result
df_result.to_csv((f"{column_names[0]}_linear.csv"))


df_test['index'] = df_test.index

from sklearn.metrics import classification_report, mean_squared_error
import numpy as np

merged_df = pd.merge(df_result, df_test, on="index")
print("merged_df",merged_df)


classification_report_var1 = classification_report(merged_df['Variable2'].round(0), merged_df['Average'])
print("classification report",classification_report_var1)

rmse_var1 = np.sqrt(mean_squared_error(merged_df['Variable1'], merged_df[column_names[0]]))
print("RMSE f"{column_names[0]}"", rmse_var1)

rmse_var2 = np.sqrt(mean_squared_error(merged_df['Variable2'], merged_df['Average']))
print("RMSE f"{column_names[1]}"", rmse_var2)

#get actual values
df_test['Average'] = df_test['Tokenized_Description1'].apply(calculate_average)
df_test
df_test1h=df_test[[column_names[0],'Tokenized_Description','Average']]
df_test1h






