!pip install --quiet pytorch-lightning
!pip install --quiet tqdm


import seaborn as sns
from pylab import rcParams
import matplotlib.pyplot as plt
from matplotlib import rc
import math
import matplotlib
import pandas as pd
import numpy as np
from tqdm.notebook import tqdm
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from collections import defaultdict
import nltk
from nltk.tokenize import word_tokenize
import shutil
from sklearn.metrics import mean_squared_error

df = pd.read_csv("./FF_Dataset_6hour_run.csv")

# Random Seed Pytorch Lightning
pl.seed_everything(42)

#for i in all column names except Description column
column_names=['LoadCell_R03','Description']

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

def calculate_average(row):
    numeric_values = [value for value in row if value is not None]
    return sum(numeric_values) / len(numeric_values) if len(numeric_values) > 0 else 0

df['Average'] = df['Tokenized_Description1'].apply(calculate_average)
features_df=df[[column_names[0],'Average']]
features_df=features_df.rename(columns={"Average": "Description"})

#train_test split
train_df = features_df[:int(0.8*(len(features_df)))]
test_df = features_df[int(0.8*(len(features_df))):]

# Normalising the Data
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scalery = scaler.fit(train_df)

train_df = pd.DataFrame(
    scalery.transform(train_df),
    index = train_df.index,
    columns = train_df.columns
)
train_df.head()

test_df = pd.DataFrame(
    scalery.transform(test_df),
    index = test_df.index,
    columns = test_df.columns
)
test_df.head()

#Generating dataframes in multiple sequences
def create_sequences(input_data: pd.DataFrame, target_columns, sequence_length=3):
    sequences = []
    data_size = len(input_data)

    for i in tqdm(range(data_size - sequence_length)):
        sequence = input_data.iloc[i:i+sequence_length]

        label_position = i + sequence_length
        labels = input_data.iloc[label_position][target_columns]

        sequences.append((sequence, labels))

    return sequences

#Creating Training and Testing Sequences
SEQUENCE_LENGTH = 120
target_columns = column_names

train_sequences = create_sequences(train_df, target_columns, sequence_length=SEQUENCE_LENGTH)
test_sequences = create_sequences(test_df, target_columns, sequence_length=SEQUENCE_LENGTH)

# To check sequence, label and shape
print("Label: ", train_sequences[0][1])
print("")
print("Sequence: ",train_sequences[0][0])
print("Sequence Shape: ",train_sequences[0][0].shape)


#Creating PyTorch Datasets
class FFDataset(Dataset):
  def __init__(self, sequences):
    self.sequences = sequences

  def __len__(self):
    return len(self.sequences)

  def __getitem__(self, idx):
    sequence, label = self.sequences[idx]
    return dict(
        sequence = torch.Tensor(sequence.to_numpy()),
        label = torch.tensor(label).float()
    )

class FFDataModule(pl.LightningDataModule):
  def __init__(
      self, train_sequences, test_sequences, batch_size = 8
  ):
    super().__init__()
    self.train_sequences = train_sequences
    self.test_sequences = test_sequences
    self.batch_size = batch_size

  def setup(self, stage=None):
    self.train_dataset = FFDataset(self.train_sequences)
    self.test_dataset = FFDataset(self.test_sequences)

  def train_dataloader(self):
    return DataLoader(
        self.train_dataset,
        batch_size = self.batch_size,
        shuffle = False,
        num_workers = 2
    )

  def val_dataloader(self):
    return DataLoader(
        self.test_dataset,
        batch_size = 1,
        shuffle = False,
        num_workers = 1
    )
  def test_dataloader(self):
    return DataLoader(
        self.test_dataset,
        batch_size = 1,
        shuffle = False,
        num_workers = 1
    )

#Model parameters
N_EPOCHS = 5
BATCH_SIZE = 64

data_module = FFDataModule(train_sequences, test_sequences, batch_size = BATCH_SIZE)
data_module.setup()

train_dataset = FFDataset(train_sequences)

# Testing the dataloader
a = iter(train_dataset)
b = next(a)
print("Sequence Shape: ", b["sequence"].shape)
print("Label: {} and Label Shape: {}".format(b["label"], b["label"].shape) )


#Model
class PredictionModel(nn.Module):
  def __init__(self, n_features, n_hidden=128, n_layers=2):
    super().__init__()

    self.n_hidden = n_hidden

    self.lstm = nn.LSTM(
        input_size = n_features,
        hidden_size = n_hidden,
        batch_first = True,
        num_layers = n_layers, # Stack LSTMs
        dropout = 0.2
    )

    self.regressor = nn.Linear(n_hidden, 2)

  def forward(self, x):
    self.lstm.flatten_parameters()  # For distrubuted training

    _, (hidden, _) = self.lstm(x)
    # We want the output from the last layer to go into the final
    # regressor linear layer
    out = hidden[-1]

    return self.regressor(out)


class Predictor(pl.LightningModule):

  def __init__(self, n_features: int):
    super().__init__()
    self.model = PredictionModel(n_features)
    self.criterion = nn.MSELoss()

  def forward(self, x, labels=None):
    output = self.model(x)

    loss = 0

    if labels is not None:
      loss = self.criterion(output, labels.unsqueeze(dim=1))

    return loss, output

  def training_step(self, batch, batch_idx):
    sequences = batch["sequence"]
    labels = batch["label"]

    loss, output = self.forward(sequences, labels)

    self.log("train_loss", loss, prog_bar=True, logger=True)
    print("train_loss", loss)
    return loss

  def validation_step(self, batch, batch_idx):
    sequences = batch["sequence"]
    labels = batch["label"]

    loss, output = self.forward(sequences, labels)

    self.log("val_loss", loss, prog_bar=True, logger=True)
    print("val_loss", loss)
    return loss

  def test_step(self, batch, batch_idx):
    sequences = batch["sequence"]
    labels = batch["label"]

    loss, output = self.forward(sequences, labels)

    self.log("test_loss", loss, prog_bar=True, logger=True)
    print("test_loss", loss)
    return loss

  def configure_optimizers(self):
    return optim.AdamW(self.model.parameters())


n_features = b["sequence"].shape[1]
n_features

model = Predictor(n_features = n_features)
n_features


for item in data_module.train_dataloader():
  print(item["sequence"].shape)
  print(item["label"].shape)
  break


shutil.rmtree('/content/lightning_logs')

checkpoint_callback = ModelCheckpoint(
    dirpath="checkpoints",
    filename="best-checkpoint",
    save_top_k = 1,
    verbose = True,
    monitor = "val_loss",
    mode = "min"
)

logger = TensorBoardLogger("lightning_logs", name = "btc-price")

early_stopping_callback = EarlyStopping(monitor = "val_loss", patience = 2)


trainer = pl.Trainer(
    logger = logger,
    #checkpoint_callback = checkpoint_callback,
    callbacks = [early_stopping_callback],
    max_epochs = N_EPOCHS,
    #gpus = 1,
   # progress_bar_refresh_rate = 30
)

trainer.fit(model, data_module)

#Testing the trained model
checkpoint_path = “Add path to .ckpt here”
trained_model = Predictor.load_from_checkpoint(
    checkpoint_path,
    n_features = n_features   # 2 in this case
)

# Freezing the model for faster predictions
trained_model.freeze()

#Getting predictions
test_dataset = FFDataset(test_sequences)
predictions_1 = []  # Predictions for the first column
predictions_2 = []  # Predictions for the second column
labels_1 = []       # Labels for the first column
labels_2 = []       # Labels for the second column

for item in tqdm(test_dataset):
    sequence = item["sequence"]
    label = item["label"]
    sequence=sequence.to(device)
    _, output = trained_model(sequence)

    # Assuming output is a tensor of shape (2,)
    pred_1, pred_2 = output.tolist()  # Splitting predictions for two columns
    predictions_1.append(pred_1)
    predictions_2.append(pred_2)

    label_1, label_2 = label.tolist()  # Splitting labels for two columns
    labels_1.append(label_1)
    labels_2.append(label_2)

test_dataset = FFDataset(test_sequences)
test_dataset
test_sequences = pd.DataFrame(test_sequences, columns=['column_name','e'])
test_sequences


# Convert lists to NumPy arrays
predictions_1 = np.array(predictions_1)
predictions_2 = np.array(predictions_2)
labels_1 = np.array(labels_1)
labels_2 = np.array(labels_2)


#Get MSE values
mse_1=mean_squared_error(labels_1, predictions_1)
rmse_1=np.sqrt(mse_1)
print(f"{column_names[0]} (RMSE): {rmse_1:.4f}")
print(f"{column_names[0]} (MSE): {mse_1:.4f}")
mse_2=mean_squared_error(labels_2, predictions_2)
rmse_2=np.sqrt(mse_2)
print(f"{column_names[1]} (RMSE): {rmse_2:.4f}")
print(f"{column_names[1]} (MSE): {mse_2:.4f}")

all_label=np.concatenate((labels_1,labels_2))
all_pred=np.concatenate((predictions_1,predictions_2))
mse_all=mean_squared_error(all_label, all_pred)
rmse_all=np.sqrt(mse_all)
print(f"All (RMSE): {rmse_all:.4f}")
print(f"All (MSE): {mse_all:.4f}")

print("predicted",predictions_1[0],predictions_2[0] )
print("labels",labels_1[0],labels_2[0])


#Doing inverse scaling
# Assuming pred1 and pred2 are your prediction arrays
pred1 = predictions_1.reshape(-1, 1)
pred2 = labels_1.reshape(-1, 1)

# Descale the predictions
descaled_pred1 = scalery.inverse_transform(np.concatenate([pred1, np.zeros_like(pred1)], axis=1))[:, 0]
descaled_pred2 = scalery.inverse_transform(np.concatenate([np.zeros_like(pred2), pred2], axis=1))[:, 1]

predictions_descaled_1 = descaled_pred1
labels_descaled_1 = descaled_pred2

print(predictions_descaled_1[:3])
print(labels_descaled_1[:3])


# Assuming pred1 and pred2 are your prediction arrays
pred_desc1 = predictions_2.reshape(-1, 1)
pred_desc2 = labels_2.reshape(-1, 1)

# Descale the predictions
descaled_pred1 = scalery.inverse_transform(np.concatenate([pred1, np.zeros_like(pred_desc1)], axis=1))[:, 0]
descaled_pred2 = scalery.inverse_transform(np.concatenate([np.zeros_like(pred_desc2), pred_desc2], axis=1))[:, 1]

predictions_descaled_2 = descaled_pred1
labels_descaled_2 = descaled_pred2

print(predictions_descaled_2[:3])
print(labels_descaled_2[:3])

vdf_pred_scaled=predictions_1.round(0)
vdf_label_scaled=labels_1.round(0)
desc_pred_scaled=predictions_2.round(0)
desc_label_scaled=labels_2.round(0)

vdf_pred_descaled=predictions_descaled_1.round(0)
vdf_label_descaled=labels_descaled_1.round(0)
desc_pred_descaled=predictions_descaled_2.round(0)
desc_label_descaled=labels_descaled_2.round(0)

print('vdf_pred_scaled ',vdf_pred_scaled)
print('vdf_label_scaled ',vdf_label_scaled)
print('desc_pred_scaled ',desc_pred_scaled)
print('desc_label_scaled ',desc_label_scaled)


print('vdf_pred_descaled ',vdf_pred_descaled)
print('vdf_label_descaled ',vdf_label_descaled)
print('desc_pred_descaled ',desc_pred_descaled)
print('desc_label_descaled ',desc_label_descaled)

#check unique value counts of predictions
unique, counts = np.unique(predictions_2.round(2), return_counts=True)
dict(zip(unique, counts))

#check unique value counts of actual values
unique, counts = np.unique(desc_label_descaled, return_counts=True)
dict(zip(unique, counts))

desc_pred_scaled = np.where(np.isclose(desc_pred_scaled, -0.0), 0, desc_pred_scaled)
desc_pred_scaled = np.where(desc_pred_scaled == 2, 9, desc_pred_scaled)
desc_pred_scaled = np.where(desc_pred_scaled == 5, 8, desc_pred_scaled)
desc_pred_scaled = np.where(desc_pred_scaled == 3, 6, desc_pred_scaled)
desc_pred_scaled = np.where(desc_pred_scaled == 1, 5, desc_pred_scaled)

unique, counts = np.unique(desc_pred_scaled, return_counts=True)
dict(zip(unique, counts))

from sklearn.metrics import  classification_report,confusion_matrix
target_names = ['0.0', '4.0','5.0','6.0','8.0','9.0']
print(classification_report(desc_label_scaled, predictions_2.round(0), target_names=target_names))

confusion_matrix(desc_label_descaled, desc_pred_scaled)

temp_desc=predictions_2.round(2)

unique, counts = np.unique(temp_desc, return_counts=True)

dict(zip(unique, counts))

x=temp_desc.size
x

#creating thresholds for different anomaly classes
for j in range(0,x):
    if temp_desc[j]< -0.18:
      temp_desc[j]=0
    elif temp_desc[j]>= -0.18 and temp_desc[j]<4.35:
      temp_desc[j]=4
    elif temp_desc[j]>=4.35 and temp_desc[j]< 4.53:
      temp_desc[j]=5
    elif temp_desc[j]>=4.53 and temp_desc[j]<4.63:
      temp_desc[j]=6
    elif temp_desc[j]>=4.63 and temp_desc[j]<4.76:
      temp_desc[j]=8
    elif temp_desc[j]>= 4.76:
      temp_desc[j]=9

#check the new distribution
unique, counts = np.unique(temp_desc, return_counts=True)
dict(zip(unique, counts))

unique, counts = np.unique(desc_label_descaled, return_counts=True)
dict(zip(unique, counts))

target_names = ['0.0', '4.0','5.0','6.0','8.0','9.0']
print(classification_report(desc_label_descaled, temp_desc, target_names=target_names))

confusion_matrix(desc_label_descaled, temp_desc)


temp_arr = np.array(temp_desc)

labels_1_temp = np.array(desc_label_descaled)

mse_1=mean_squared_error(labels_1_temp, temp_arr)
rmse_1=np.sqrt(mse_1)
print(f"{column_names[1]} (RMSE): {rmse_1:.4f}")
print(f"{column_names[1]} (MSE): {mse_1:.4f}")
par_arr = np.array(vdf_pred_descaled)

labels_1_pred = np.array(desc_label_descaled)


all_label=np.concatenate((labels_1,labels_1_temp))
all_pred=np.concatenate((predictions_1,temp_arr))
mse_all=mean_squared_error(all_label, all_pred)
rmse_all=np.sqrt(mse_all)
print(f"All (RMSE): {rmse_all:.4f}")
print(f"All (MSE): {mse_all:.4f}")

actual=desc_label_descaled.round(0)

unique, counts = np.unique(actual, return_counts=True)
dict(zip(unique, counts))

temp=pd.DataFrame(desc_pred_scaled,columns =['prediction'])
actual=pd.DataFrame(actual,columns =['actual'])

predictions_scaled_2=pd.DataFrame(vdf_pred_scaled,columns =[f'{column_names[0]}'])

temp=pd.DataFrame(temp_desc,columns =['prediction'])

predictions_descaled_2=pd.DataFrame(predictions_descaled_1.round(6),columns =[f'{column_names[0]}'])

merged_df = pd.merge(predictions_descaled_2, temp, left_index=True, right_index=True)

merged_df.to_csv((f"{column_names[0]}_lstm.csv"))


vdf = pd.read_csv(f"{column_names[0]}_lstm.csv")
