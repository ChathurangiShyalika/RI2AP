!pip install --quiet pytorch-lightning
!pip install --quiet tqdm

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from torch.optim.lr_scheduler import ReduceLROnPlateau
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
import nltk
from nltk.tokenize import word_tokenize
import shutil
from sklearn.metrics import mean_squared_error

df = pd.read_csv("./FF_Dataset_6hour_run.csv")

#change column names accordingly

#for i in all column names except Description column
column_names=['VFD2','Description']

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

df=features_df
df['VFD2'] = df['VFD2'].astype(float)
df['Description'] = df['Description'].astype(float)


df=df[['VFD2','Description']]

split_index = int(0.8 * len(df))

# Split the data into training and validation sets
df_train = df.iloc[:split_index]
df_test = df.iloc[split_index:]

spots_train = df_train[['VFD2','Description']].to_numpy()
spots_test = df_test[['VFD2','Description']].to_numpy()
spots_train[:2]

# Select the columns and scale them separately
scaler = StandardScaler()

spots_train = scaler.fit_transform(df_train[['VFD2','Description']])
spots_test = scaler.transform(df_test[['VFD2','Description']])

# Convert the arrays to lists
spots_train = spots_train.tolist()
spots_test = spots_test.tolist()
spots_train[:2]

#Making sequences of data
SEQUENCE_SIZE = 10

def to_sequences(seq_size, obs):
    x = []
    y = []
    for i in range(len(obs) - seq_size):
        window = obs[i:(i + seq_size)]
        after_window = obs[i + seq_size]
        x.append(window)
        y.append(after_window)
    return torch.tensor(x, dtype=torch.float32).view(-1, seq_size, obs.shape[1]), torch.tensor(y, dtype=torch.float32).view(-1, obs.shape[1])

# Assuming spots_train is a numpy array containing scaled values for both columns
x_train, y_train = to_sequences(SEQUENCE_SIZE, np.array(spots_train))
x_test, y_test = to_sequences(SEQUENCE_SIZE, np.array(spots_test))

# Setup data loaders for batch
train_dataset = TensorDataset(x_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

test_dataset = TensorDataset(x_test, y_test)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Positional Encoding for Transformer
class PositionalEncoding(pl.LightningModule):
    def __init__(self, d_model, dropout=0.1, max_len=500):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

device = (
    "mps"
    if getattr(torch, "has_mps", False)
    else "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

#Transformer model
class TimeSeriesTransformer(pl.LightningModule):
    def __init__(self, input_dim=2, d_model=512, nhead=4, num_layers=2, dropout=0.2):
        super().__init__()

        self.encoder = nn.Linear(input_dim, d_model)
        # self.pos_encoder = PositionalEncoding(d_model, dropout)
        self.pos_encoder = nn.Embedding(len(df_train), d_model)
        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        self.decoder = nn.Linear(d_model, 2)

        self.criterion = nn.MSELoss()
        self.learning_rate = 0.01

    def forward(self, x):
        x = self.encoder(x)
        l = torch.arange(len(x[1]))
        l = l.to(device)



        x = [x[i]+self.pos_encoder(l) for i in range(x.size()[0])]
        x=torch.stack(x)


        x = self.transformer_encoder(x)
        x = self.decoder(x[:, -1, :])
        return x

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters())
        scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=3, verbose=True)
        return {'optimizer': optimizer, 'lr_scheduler': scheduler, 'monitor': 'val_loss'}

    def training_step(self, batch, batch_idx):
        x_batch, y_batch = batch
        outputs = self(x_batch)
        loss = self.criterion(outputs, y_batch)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x_batch, y_batch = batch
        outputs = self(x_batch)
        loss = self.criterion(outputs, y_batch)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss


model = TimeSeriesTransformer()


shutil.rmtree('/content/lightning_logs')

# Setup model checkpoint callback
checkpoint_callback = pl.callbacks.ModelCheckpoint(
    dirpath="lightning_logs",
    filename="best-checkpoint",
    save_top_k = 1,
    verbose = True,
    monitor = "val_loss",
    mode = "min"
    )

logger = TensorBoardLogger("lightning_logs", name = "ts_model")
early_stopping_callback = EarlyStopping(monitor = "val_loss", patience = 2)

# Setup PyTorch Lightning Trainer
trainer = pl.Trainer(
    max_epochs=5,
    logger = logger,
    callbacks=[early_stopping_callback],
    #gpus=1 if torch.cuda.is_available() else None
    )

trainer.fit(model, train_loader, test_loader)

# Evaluation
checkpoint_path=“Add path to .ckpt here”
model = TimeSeriesTransformer.load_from_checkpoint(checkpoint_path)

model.eval()
predictions = []
with torch.no_grad():
    for batch in test_loader:
        x_batch, y_batch = batch
        x_batch = x_batch.to(device)
        outputs = model(x_batch)
        predictions.extend(outputs.tolist())

# Convert predictions and labels back to their original scale
predictions = scaler.inverse_transform(np.array(predictions))
labels = scaler.inverse_transform(y_test.numpy())

# Calculate RMSE considering 2D structure
rmse = np.sqrt(np.mean((predictions - labels) ** 2))
print(f"Score (RMSE): {rmse:.4f}")

unique, counts = np.unique(labels_2, return_counts=True)
dict(zip(unique, counts))

temp=predictions_2
x=temp.size

for j in range(0,x):
    if temp[j]<0.33219855:
      temp[j]=0
    elif temp[j]>=0.33219855 and temp[j]<0.33219856:
      temp[j]=4
    elif temp[j]>=0.33219856 and temp[j]<0.3321986:
      temp[j]=5
    elif temp[j]>=0.3321986 and temp[j]<0.3321987:
      temp[j]=6
    elif temp[j]>=0.3321987 and temp[j]<0.332199:
      temp[j]=8
    elif temp[j]>=0.332199:
      temp[j]=9

unique, counts = np.unique(temp, return_counts=True)
dict(zip(unique, counts))

actual=labels_2
actual_size=actual.size

for j in range(0,actual_size):
    if actual[j]<2.25:
      actual[j]=0
    elif actual[j]>=2.25 and actual[j]<5.0:
      actual[j]=4
    elif actual[j]>=5.0 and actual[j]<5.875:
      actual[j]=5
    elif actual[j]>=5.875 and actual[j]<7.125:
      actual[j]=6
    elif actual[j]>=7.125 and actual[j]<8.5:
      actual[j]=8
    elif actual[j]>=8.5:
      actual[j]=9

unique, counts = np.unique(actual, return_counts=True)
dict(zip(unique, counts))

temp=pd.DataFrame(temp,columns =['prediction'])
actual=pd.DataFrame(actual,columns =['actual'])

target_names = ['0', '4','5','6','8','9']
print(classification_report(actual, temp, target_names=target_names))

confusion_matrix(actual, temp)

#Get the MSE values
mse_1=mean_squared_error(labels_1, predictions_1)
rmse_1=np.sqrt(mse_1)
print(f"{column_names[0]} (RMSE): {rmse_1:.4f}")
mse_2=mean_squared_error(labels_2, predictions_2)
rmse_2=np.sqrt(mse_2)
print(f"{column_names[1]} (RMSE): {rmse_2:.4f}")


merged_df.to_csv(f"{column_names[0]}_transformer.csv")

