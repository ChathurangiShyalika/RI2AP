{\rtf1\ansi\ansicpg1252\cocoartf2636
\cocoatextscaling0\cocoaplatform0{\fonttbl\f0\fmodern\fcharset0 Courier;}
{\colortbl;\red255\green255\blue255;\red113\green184\blue255;\red23\green23\blue23;\red202\green202\blue202;
\red183\green111\blue179;\red212\green212\blue212;\red183\green111\blue179;\red23\green23\blue23;\red202\green202\blue202;
\red194\green126\blue101;\red89\green156\blue62;\red70\green137\blue204;\red67\green192\blue160;\red167\green197\blue152;
\red212\green214\blue154;\red140\green211\blue254;}
{\*\expandedcolortbl;;\cssrgb\c50980\c77647\c100000;\cssrgb\c11765\c11765\c11765;\cssrgb\c83137\c83137\c83137;
\cssrgb\c77255\c52549\c75294;\cssrgb\c86275\c86275\c86275;\cssrgb\c77255\c52549\c75294;\cssrgb\c11765\c11765\c11765;\cssrgb\c83137\c83137\c83137;
\cssrgb\c80784\c56863\c47059;\cssrgb\c41569\c66275\c30980;\cssrgb\c33725\c61176\c83922;\cssrgb\c30588\c78824\c69020;\cssrgb\c70980\c80784\c65882;
\cssrgb\c86275\c86275\c66667;\cssrgb\c61176\c86275\c99608;}
\margl1440\margr1440\vieww11520\viewh8400\viewkind0
\deftab720
\pard\pardeftab720\partightenfactor0

\f0\fs28 \cf2 \cb3 \expnd0\expndtw0\kerning0
\outl0\strokewidth0 \strokec2 !\cf4 \strokec4 pip install --quiet pytorch-lightning\cb1 \
\cf2 \cb3 \strokec2 !\cf4 \strokec4 pip install --quiet tqdm\cb1 \
\
\pard\pardeftab720\partightenfactor0
\cf5 \cb3 \strokec5 import\cf4 \cb3 \strokec4  torch\cb1 \
\cf5 \cb3 \strokec5 import\cf4 \cb3 \strokec4  torch.nn \cf5 \cb3 \strokec5 as\cf4 \cb3 \strokec4  nn\cb1 \
\cf5 \cb3 \strokec5 import\cf4 \cb3 \strokec4  numpy \cf5 \cb3 \strokec5 as\cf4 \cb3 \strokec4  np\cb1 \
\cf5 \cb3 \strokec5 import\cf4 \cb3 \strokec4  pandas \cf5 \cb3 \strokec5 as\cf4 \cb3 \strokec4  pd\cb1 \
\cf5 \cb3 \strokec5 from\cf4 \cb3 \strokec4  torch.utils.data \cf5 \cb3 \strokec5 import\cf4 \cb3 \strokec4  DataLoader\cf6 \cb3 \strokec6 ,\cf4 \cb3 \strokec4  TensorDataset\cb1 \
\cf5 \cb3 \strokec5 from\cf4 \cb3 \strokec4  sklearn.preprocessing \cf5 \cb3 \strokec5 import\cf4 \cb3 \strokec4  StandardScaler\cb1 \
\cf5 \cb3 \strokec5 from\cf4 \cb3 \strokec4  torch.optim.lr_scheduler \cf5 \cb3 \strokec5 import\cf4 \cb3 \strokec4  ReduceLROnPlateau\cb1 \
\cf5 \cb3 \strokec5 import\cf4 \cb3 \strokec4  pytorch_lightning \cf5 \cb3 \strokec5 as\cf4 \cb3 \strokec4  pl\cb1 \
\cf5 \cb3 \strokec5 from\cf4 \cb3 \strokec4  pytorch_lightning.callbacks \cf5 \cb3 \strokec5 import\cf4 \cb3 \strokec4  ModelCheckpoint\cf6 \cb3 \strokec6 ,\cf4 \cb3 \strokec4  EarlyStopping\cb1 \
\cf5 \cb3 \strokec5 from\cf4 \cb3 \strokec4  pytorch_lightning.loggers \cf5 \cb3 \strokec5 import\cf4 \cb3 \strokec4  TensorBoardLogger\cb1 \
\cf5 \cb3 \strokec5 import\cf4 \cb3 \strokec4  nltk\cb1 \
\cf5 \cb3 \strokec5 from\cf4 \cb3 \strokec4  nltk.tokenize \cf5 \cb3 \strokec5 import\cf4 \cb3 \strokec4  word_tokenize\cb1 \
\pard\pardeftab720\partightenfactor0
\cf7 \cb8 \outl0\strokewidth0 import\cf9  shutil\cb1 \
\cf7 \cb8 from\cf9  sklearn.metrics \cf7 import\cf9  mean_squared_error\cb1 \
\pard\pardeftab720\partightenfactor0
\cf4 \outl0\strokewidth0 \strokec4 \
\pard\pardeftab720\partightenfactor0
\cf4 \cb3 df = pd.read_csv\cf6 \cb3 \strokec6 (\cf10 \strokec10 "FF_Dataset_6hour_run.csv"\cf6 \strokec6 )\cf4 \cb1 \strokec4 \
\
\pard\pardeftab720\partightenfactor0
\cf11 \cb3 \strokec11 #change column names accordingly\cf4 \cb1 \strokec4 \
\
\cf11 \cb3 \strokec11 #for i in all column names except Description column\cf4 \cb1 \strokec4 \
\pard\pardeftab720\partightenfactor0
\cf4 \cb3 column_names=\cf6 \cb3 \strokec6 [\cf10 \strokec10 'VFD2'\cf6 \strokec6 ,\cf10 \strokec10 'Description'\cf6 \strokec6 ]\cf4 \cb1 \strokec4 \
\
\cb3 df=df\cf6 \cb3 \strokec6 [\cf4 \cb3 \strokec4 column_names\cf6 \cb3 \strokec6 ]\cf4 \cb1 \strokec4 \
\cb3 nltk.download\cf6 \cb3 \strokec6 (\cf10 \strokec10 'punkt'\cf6 \strokec6 )\cf4 \cb3 \strokec4   \cf11 \cb3 \strokec11 # Download the NLTK tokenizer data (if not downloaded)\cf4 \cb1 \strokec4 \
\cb3 df\cf6 \cb3 \strokec6 [\cf10 \strokec10 'Tokenized_Description'\cf6 \strokec6 ]\cf4 \cb3 \strokec4  = df\cf6 \cb3 \strokec6 [\cf10 \strokec10 'Description'\cf6 \strokec6 ]\cf4 \cb3 \strokec4 .apply\cf6 \cb3 \strokec6 (\cf12 \cb3 \strokec12 lambda\cf4 \cb3 \strokec4  x\cf6 \cb3 \strokec6 :\cf4 \cb3 \strokec4  word_tokenize\cf6 \cb3 \strokec6 (\cf13 \cb3 \strokec13 str\cf6 \cb3 \strokec6 (\cf4 \cb3 \strokec4 x\cf6 \cb3 \strokec6 ))\cf4 \cb3 \strokec4  \cf5 \cb3 \strokec5 if\cf4 \cb3 \strokec4  pd.notnull\cf6 \cb3 \strokec6 (\cf4 \cb3 \strokec4 x\cf6 \cb3 \strokec6 )\cf4 \cb3 \strokec4  \cf5 \cb3 \strokec5 else\cf4 \cb3 \strokec4  \cf6 \cb3 \strokec6 [])\cf4 \cb1 \strokec4 \
\
\cb3 result = df\cf6 \cb3 \strokec6 [[\cf4 \cb3 \strokec4 column_names\cf6 \cb3 \strokec6 [\cf14 \cb3 \strokec14 0\cf6 \cb3 \strokec6 ],\cf4 \cb3 \strokec4  \cf10 \cb3 \strokec10 'Tokenized_Description'\cf6 \strokec6 ]]\cf4 \cb3 \strokec4 .values.tolist\cf6 \cb3 \strokec6 ()\cf4 \cb1 \strokec4 \
\
\cb3 all_tokens = \cf13 \cb3 \strokec13 set\cf6 \cb3 \strokec6 ()\cf4 \cb1 \strokec4 \
\cb3 _ = df\cf6 \cb3 \strokec6 [\cf10 \strokec10 'Tokenized_Description'\cf6 \strokec6 ]\cf4 \cb3 \strokec4 .apply\cf6 \cb3 \strokec6 (\cf12 \cb3 \strokec12 lambda\cf4 \cb3 \strokec4  x\cf6 \cb3 \strokec6 :\cf4 \cb3 \strokec4  all_tokens.update\cf6 \cb3 \strokec6 (\cf4 \cb3 \strokec4 x\cf6 \cb3 \strokec6 )\cf4 \cb3 \strokec4  \cf5 \cb3 \strokec5 if\cf4 \cb3 \strokec4  x \cf5 \cb3 \strokec5 else\cf4 \cb3 \strokec4  all_tokens.add\cf6 \cb3 \strokec6 (\cf12 \cb3 \strokec12 None\cf6 \cb3 \strokec6 ))\cf4 \cb1 \strokec4 \
\
\pard\pardeftab720\partightenfactor0
\cf11 \cb3 \strokec11 # List of all different tokens\cf4 \cb1 \strokec4 \
\pard\pardeftab720\partightenfactor0
\cf4 \cb3 different_tokens = \cf13 \cb3 \strokec13 list\cf6 \cb3 \strokec6 (\cf4 \cb3 \strokec4 all_tokens\cf6 \cb3 \strokec6 )\cf4 \cb1 \strokec4 \
\
\cb3 tokens = \cf6 \cb3 \strokec6 \{\cf10 \strokec10 'None'\cf6 \strokec6 :\cf14 \cb3 \strokec14 0.0\cf6 \cb3 \strokec6 ,\cf10 \strokec10 'Nose'\cf6 \strokec6 :\cf14 \cb3 \strokec14 1.0\cf6 \cb3 \strokec6 ,\cf4 \cb3 \strokec4  \cf10 \cb3 \strokec10 'nose'\cf6 \strokec6 :\cf4 \cb3 \strokec4  \cf14 \cb3 \strokec14 2.0\cf6 \cb3 \strokec6 ,\cf4 \cb3 \strokec4  \cf10 \cb3 \strokec10 'Removed'\cf6 \strokec6 :\cf4 \cb3 \strokec4  \cf14 \cb3 \strokec14 3.0\cf6 \cb3 \strokec6 ,\cf4 \cb3 \strokec4  \cf10 \cb3 \strokec10 'crashed'\cf6 \strokec6 :\cf4 \cb3 \strokec4  \cf14 \cb3 \strokec14 4.0\cf6 \cb3 \strokec6 ,\cf4 \cb3 \strokec4  \cf10 \cb3 \strokec10 'R03'\cf6 \strokec6 :\cf4 \cb3 \strokec4  \cf14 \cb3 \strokec14 5.0\cf6 \cb3 \strokec6 ,\cf4 \cb3 \strokec4  \cf10 \cb3 \strokec10 'Nosecone'\cf6 \strokec6 :\cf4 \cb3 \strokec4  \cf14 \cb3 \strokec14 6.0\cf6 \cb3 \strokec6 ,\cf4 \cb3 \strokec4  \cf10 \cb3 \strokec10 'BothBodies'\cf6 \strokec6 :\cf4 \cb3 \strokec4  \cf14 \cb3 \strokec14 7.0\cf6 \cb3 \strokec6 ,\cf4 \cb3 \strokec4  \cf10 \cb3 \strokec10 'R04'\cf6 \strokec6 :\cf4 \cb3 \strokec4  \cf14 \cb3 \strokec14 8.0\cf6 \cb3 \strokec6 ,\cf4 \cb3 \strokec4  \cf10 \cb3 \strokec10 'Door2_TimedOut'\cf6 \strokec6 :\cf4 \cb3 \strokec4  \cf14 \cb3 \strokec14 9.0\cf6 \cb3 \strokec6 ,\cf4 \cb3 \strokec4  \cf10 \cb3 \strokec10 'TopBody'\cf6 \strokec6 :\cf4 \cb3 \strokec4  \cf14 \cb3 \strokec14 10.0\cf6 \cb3 \strokec6 ,\cf4 \cb3 \strokec4  \cf10 \cb3 \strokec10 'and'\cf6 \strokec6 :\cf4 \cb3 \strokec4  \cf14 \cb3 \strokec14 11.0\cf6 \cb3 \strokec6 ,\cf4 \cb3 \strokec4  \cf10 \cb3 \strokec10 'ESTOPPED'\cf6 \strokec6 :\cf4 \cb3 \strokec4  \cf14 \cb3 \strokec14 12.0\cf6 \cb3 \strokec6 ,\cf4 \cb3 \strokec4  \cf10 \cb3 \strokec10 'Body2'\cf6 \strokec6 :\cf4 \cb3 \strokec4  \cf14 \cb3 \strokec14 13.0\cf6 \cb3 \strokec6 ,\cf4 \cb3 \strokec4  \cf10 \cb3 \strokec10 'tail'\cf6 \strokec6 :\cf4 \cb3 \strokec4  \cf14 \cb3 \strokec14 14.0\cf6 \cb3 \strokec6 \}\cf4 \cb1 \strokec4 \
\
\pard\pardeftab720\partightenfactor0
\cf12 \cb3 \strokec12 def\cf4 \cb3 \strokec4  \cf15 \cb3 \strokec15 replace_with_numeric\cf4 \cb3 \strokec4 (\cf16 \cb3 \strokec16 tokens_dict\cf4 \cb3 \strokec4 , \cf16 \cb3 \strokec16 tokens_list\cf4 \cb3 \strokec4 )\cf6 \cb3 \strokec6 :\cf4 \cb1 \strokec4 \
\pard\pardeftab720\partightenfactor0
\cf4 \cb3     \cf5 \cb3 \strokec5 return\cf4 \cb3 \strokec4  \cf6 \cb3 \strokec6 [\cf4 \cb3 \strokec4 tokens_dict\cf6 \cb3 \strokec6 [\cf4 \cb3 \strokec4 token\cf6 \cb3 \strokec6 ]\cf4 \cb3 \strokec4  \cf5 \cb3 \strokec5 if\cf4 \cb3 \strokec4  token \cf2 \strokec2 in\cf4 \strokec4  tokens_dict \cf5 \cb3 \strokec5 else\cf4 \cb3 \strokec4  \cf12 \cb3 \strokec12 None\cf4 \cb3 \strokec4  \cf5 \cb3 \strokec5 for\cf4 \cb3 \strokec4  token \cf2 \strokec2 in\cf4 \strokec4  tokens_list\cf6 \cb3 \strokec6 ]\cf4 \cb1 \strokec4 \
\
\cb3 df\cf6 \cb3 \strokec6 [\cf10 \strokec10 'Tokenized_Description1'\cf6 \strokec6 ]\cf4 \cb3 \strokec4  = df\cf6 \cb3 \strokec6 [\cf10 \strokec10 'Tokenized_Description'\cf6 \strokec6 ]\cf4 \cb3 \strokec4 .apply\cf6 \cb3 \strokec6 (\cf12 \cb3 \strokec12 lambda\cf4 \cb3 \strokec4  x\cf6 \cb3 \strokec6 :\cf4 \cb3 \strokec4  replace_with_numeric\cf6 \cb3 \strokec6 (\cf4 \cb3 \strokec4 tokens\cf6 \cb3 \strokec6 ,\cf4 \cb3 \strokec4  x\cf6 \cb3 \strokec6 ))\cf4 \cb1 \strokec4 \
\
\pard\pardeftab720\partightenfactor0
\cf12 \cb3 \strokec12 def\cf4 \cb3 \strokec4  \cf15 \cb3 \strokec15 calculate_average\cf4 \cb3 \strokec4 (\cf16 \cb3 \strokec16 row\cf4 \cb3 \strokec4 )\cf6 \cb3 \strokec6 :\cf4 \cb1 \strokec4 \
\pard\pardeftab720\partightenfactor0
\cf4 \cb3     numeric_values = \cf6 \cb3 \strokec6 [\cf4 \cb3 \strokec4 value \cf5 \cb3 \strokec5 for\cf4 \cb3 \strokec4  value \cf2 \strokec2 in\cf4 \strokec4  row \cf5 \cb3 \strokec5 if\cf4 \cb3 \strokec4  value \cf2 \strokec2 is\cf4 \strokec4  \cf2 \strokec2 not\cf4 \strokec4  \cf12 \cb3 \strokec12 None\cf6 \cb3 \strokec6 ]\cf4 \cb1 \strokec4 \
\cb3     \cf5 \cb3 \strokec5 return\cf4 \cb3 \strokec4  \cf15 \cb3 \strokec15 sum\cf6 \cb3 \strokec6 (\cf4 \cb3 \strokec4 numeric_values\cf6 \cb3 \strokec6 )\cf4 \cb3 \strokec4  / \cf15 \cb3 \strokec15 len\cf6 \cb3 \strokec6 (\cf4 \cb3 \strokec4 numeric_values\cf6 \cb3 \strokec6 )\cf4 \cb3 \strokec4  \cf5 \cb3 \strokec5 if\cf4 \cb3 \strokec4  \cf15 \cb3 \strokec15 len\cf6 \cb3 \strokec6 (\cf4 \cb3 \strokec4 numeric_values\cf6 \cb3 \strokec6 )\cf4 \cb3 \strokec4  > \cf14 \cb3 \strokec14 0\cf4 \cb3 \strokec4  \cf5 \cb3 \strokec5 else\cf4 \cb3 \strokec4  \cf14 \cb3 \strokec14 0\cf4 \cb1 \strokec4 \
\
\cb3 df\cf6 \cb3 \strokec6 [\cf10 \strokec10 'Average'\cf6 \strokec6 ]\cf4 \cb3 \strokec4  = df\cf6 \cb3 \strokec6 [\cf10 \strokec10 'Tokenized_Description1'\cf6 \strokec6 ]\cf4 \cb3 \strokec4 .apply\cf6 \cb3 \strokec6 (\cf4 \cb3 \strokec4 calculate_average\cf6 \cb3 \strokec6 )\cf4 \cb1 \strokec4 \
\cb3 features_df=df\cf6 \cb3 \strokec6 [[\cf4 \cb3 \strokec4 column_names\cf6 \cb3 \strokec6 [\cf14 \cb3 \strokec14 0\cf6 \cb3 \strokec6 ],\cf10 \strokec10 'Average'\cf6 \strokec6 ]]\cf4 \cb1 \strokec4 \
\cb3 features_df=features_df.rename\cf6 \cb3 \strokec6 (\cf4 \cb3 \strokec4 columns=\cf6 \cb3 \strokec6 \{\cf10 \strokec10 "Average"\cf6 \strokec6 :\cf4 \cb3 \strokec4  \cf10 \cb3 \strokec10 "Description"\cf6 \strokec6 \})\cf4 \cb1 \strokec4 \
\cb3 \
df=features_df\cb1 \
\cb3 df\cf6 \cb3 \strokec6 [\cf10 \strokec10 'VFD2'\cf6 \strokec6 ]\cf4 \cb3 \strokec4  = df\cf6 \cb3 \strokec6 [\cf10 \strokec10 'VFD2'\cf6 \strokec6 ]\cf4 \cb3 \strokec4 .astype\cf6 \cb3 \strokec6 (\cf13 \cb3 \strokec13 float\cf6 \cb3 \strokec6 )\cf4 \cb1 \strokec4 \
\cb3 df\cf6 \cb3 \strokec6 [\cf10 \strokec10 'Description'\cf6 \strokec6 ]\cf4 \cb3 \strokec4  = df\cf6 \cb3 \strokec6 [\cf10 \strokec10 'Description'\cf6 \strokec6 ]\cf4 \cb3 \strokec4 .astype\cf6 \cb3 \strokec6 (\cf13 \cb3 \strokec13 float\cf6 \cb3 \strokec6 )\cf4 \cb1 \strokec4 \
\
\
\cb3 df=df\cf6 \cb3 \strokec6 [[\cf10 \strokec10 'VFD2'\cf6 \strokec6 ,\cf10 \strokec10 'Description'\cf6 \strokec6 ]]\
\pard\pardeftab720\partightenfactor0
\cf4 \cb1 \strokec4 \
\pard\pardeftab720\partightenfactor0
\cf4 \cb3 split_index = \cf13 \cb3 \strokec13 int\cf6 \cb3 \strokec6 (\cf14 \cb3 \strokec14 0.8\cf4 \cb3 \strokec4  * \cf15 \cb3 \strokec15 len\cf6 \cb3 \strokec6 (\cf4 \cb3 \strokec4 df\cf6 \cb3 \strokec6 ))\cf4 \cb1 \strokec4 \
\
\pard\pardeftab720\partightenfactor0
\cf11 \cb3 \strokec11 # Split the data into training and validation sets\cf4 \cb1 \strokec4 \
\pard\pardeftab720\partightenfactor0
\cf4 \cb3 df_train = df.iloc\cf6 \cb3 \strokec6 [:\cf4 \cb3 \strokec4 split_index\cf6 \cb3 \strokec6 ]\cf4 \cb1 \strokec4 \
\cb3 df_test = df.iloc\cf6 \cb3 \strokec6 [\cf4 \cb3 \strokec4 split_index\cf6 \cb3 \strokec6 :]\
\
\cf4 \cb3 \strokec4 spots_train = df_train\cf6 \cb3 \strokec6 [[\cf10 \strokec10 'VFD2'\cf6 \strokec6 ,\cf10 \strokec10 'Description'\cf6 \strokec6 ]]\cf4 \cb3 \strokec4 .to_numpy\cf6 \cb3 \strokec6 ()\cf4 \cb1 \strokec4 \
\cb3 spots_test = df_test\cf6 \cb3 \strokec6 [[\cf10 \strokec10 'VFD2'\cf6 \strokec6 ,\cf10 \strokec10 'Description'\cf6 \strokec6 ]]\cf4 \cb3 \strokec4 .to_numpy\cf6 \cb3 \strokec6 ()\cf4 \cb1 \strokec4 \
\cb3 spots_train\cf6 \cb3 \strokec6 [:\cf14 \cb3 \strokec14 2\cf6 \cb3 \strokec6 ]\cf4 \cb1 \strokec4 \
\
\pard\pardeftab720\partightenfactor0
\cf11 \cb3 \strokec11 # Select the columns and scale them separately\cf4 \cb1 \strokec4 \
\pard\pardeftab720\partightenfactor0
\cf4 \cb3 scaler = StandardScaler\cf6 \cb3 \strokec6 ()\cf4 \cb1 \strokec4 \
\
\cb3 spots_train = scaler.fit_transform\cf6 \cb3 \strokec6 (\cf4 \cb3 \strokec4 df_train\cf6 \cb3 \strokec6 [[\cf10 \strokec10 'VFD2'\cf6 \strokec6 ,\cf10 \strokec10 'Description'\cf6 \strokec6 ]])\cf4 \cb1 \strokec4 \
\cb3 spots_test = scaler.transform\cf6 \cb3 \strokec6 (\cf4 \cb3 \strokec4 df_test\cf6 \cb3 \strokec6 [[\cf10 \strokec10 'VFD2'\cf6 \strokec6 ,\cf10 \strokec10 'Description'\cf6 \strokec6 ]])\cf4 \cb1 \strokec4 \
\
\pard\pardeftab720\partightenfactor0
\cf11 \cb3 \strokec11 # Convert the arrays to lists\cf4 \cb1 \strokec4 \
\pard\pardeftab720\partightenfactor0
\cf4 \cb3 spots_train = spots_train.tolist\cf6 \cb3 \strokec6 ()\cf4 \cb1 \strokec4 \
\cb3 spots_test = spots_test.tolist\cf6 \cb3 \strokec6 ()\cf4 \cb1 \strokec4 \
\cb3 spots_train\cf6 \cb3 \strokec6 [:\cf14 \cb3 \strokec14 2\cf6 \cb3 \strokec6 ]\cf4 \cb1 \strokec4 \
\
#Making sequences of data\
\cb3 SEQUENCE_SIZE = \cf14 \cb3 \strokec14 10\cf4 \cb1 \strokec4 \
\
\pard\pardeftab720\partightenfactor0
\cf12 \cb3 \strokec12 def\cf4 \cb3 \strokec4  \cf15 \cb3 \strokec15 to_sequences\cf4 \cb3 \strokec4 (\cf16 \cb3 \strokec16 seq_size\cf4 \cb3 \strokec4 , \cf16 \cb3 \strokec16 obs\cf4 \cb3 \strokec4 )\cf6 \cb3 \strokec6 :\cf4 \cb1 \strokec4 \
\pard\pardeftab720\partightenfactor0
\cf4 \cb3     x = \cf6 \cb3 \strokec6 []\cf4 \cb1 \strokec4 \
\cb3     y = \cf6 \cb3 \strokec6 []\cf4 \cb1 \strokec4 \
\cb3     \cf5 \cb3 \strokec5 for\cf4 \cb3 \strokec4  i \cf2 \strokec2 in\cf4 \strokec4  \cf15 \cb3 \strokec15 range\cf6 \cb3 \strokec6 (\cf15 \cb3 \strokec15 len\cf6 \cb3 \strokec6 (\cf4 \cb3 \strokec4 obs\cf6 \cb3 \strokec6 )\cf4 \cb3 \strokec4  - seq_size\cf6 \cb3 \strokec6 ):\cf4 \cb1 \strokec4 \
\cb3         window = obs\cf6 \cb3 \strokec6 [\cf4 \cb3 \strokec4 i\cf6 \cb3 \strokec6 :(\cf4 \cb3 \strokec4 i + seq_size\cf6 \cb3 \strokec6 )]\cf4 \cb1 \strokec4 \
\cb3         after_window = obs\cf6 \cb3 \strokec6 [\cf4 \cb3 \strokec4 i + seq_size\cf6 \cb3 \strokec6 ]\cf4 \cb1 \strokec4 \
\cb3         x.append\cf6 \cb3 \strokec6 (\cf4 \cb3 \strokec4 window\cf6 \cb3 \strokec6 )\cf4 \cb1 \strokec4 \
\cb3         y.append\cf6 \cb3 \strokec6 (\cf4 \cb3 \strokec4 after_window\cf6 \cb3 \strokec6 )\cf4 \cb1 \strokec4 \
\cb3     \cf5 \cb3 \strokec5 return\cf4 \cb3 \strokec4  torch.tensor\cf6 \cb3 \strokec6 (\cf4 \cb3 \strokec4 x\cf6 \cb3 \strokec6 ,\cf4 \cb3 \strokec4  dtype=torch.float32\cf6 \cb3 \strokec6 )\cf4 \cb3 \strokec4 .view\cf6 \cb3 \strokec6 (\cf14 \cb3 \strokec14 -1\cf6 \cb3 \strokec6 ,\cf4 \cb3 \strokec4  seq_size\cf6 \cb3 \strokec6 ,\cf4 \cb3 \strokec4  obs.shape\cf6 \cb3 \strokec6 [\cf14 \cb3 \strokec14 1\cf6 \cb3 \strokec6 ]),\cf4 \cb3 \strokec4  torch.tensor\cf6 \cb3 \strokec6 (\cf4 \cb3 \strokec4 y\cf6 \cb3 \strokec6 ,\cf4 \cb3 \strokec4  dtype=torch.float32\cf6 \cb3 \strokec6 )\cf4 \cb3 \strokec4 .view\cf6 \cb3 \strokec6 (\cf14 \cb3 \strokec14 -1\cf6 \cb3 \strokec6 ,\cf4 \cb3 \strokec4  obs.shape\cf6 \cb3 \strokec6 [\cf14 \cb3 \strokec14 1\cf6 \cb3 \strokec6 ])\cf4 \cb1 \strokec4 \
\
\pard\pardeftab720\partightenfactor0
\cf11 \cb3 \strokec11 # Assuming spots_train is a numpy array containing scaled values for both columns\cf4 \cb1 \strokec4 \
\pard\pardeftab720\partightenfactor0
\cf4 \cb3 x_train\cf6 \cb3 \strokec6 ,\cf4 \cb3 \strokec4  y_train = to_sequences\cf6 \cb3 \strokec6 (\cf4 \cb3 \strokec4 SEQUENCE_SIZE\cf6 \cb3 \strokec6 ,\cf4 \cb3 \strokec4  np.array\cf6 \cb3 \strokec6 (\cf4 \cb3 \strokec4 spots_train\cf6 \cb3 \strokec6 ))\cf4 \cb1 \strokec4 \
\cb3 x_test\cf6 \cb3 \strokec6 ,\cf4 \cb3 \strokec4  y_test = to_sequences\cf6 \cb3 \strokec6 (\cf4 \cb3 \strokec4 SEQUENCE_SIZE\cf6 \cb3 \strokec6 ,\cf4 \cb3 \strokec4  np.array\cf6 \cb3 \strokec6 (\cf4 \cb3 \strokec4 spots_test\cf6 \cb3 \strokec6 ))\cf4 \cb1 \strokec4 \
\
\pard\pardeftab720\partightenfactor0
\cf11 \cb3 \strokec11 # Setup data loaders for batch\cf4 \cb1 \strokec4 \
\pard\pardeftab720\partightenfactor0
\cf4 \cb3 train_dataset = TensorDataset\cf6 \cb3 \strokec6 (\cf4 \cb3 \strokec4 x_train\cf6 \cb3 \strokec6 ,\cf4 \cb3 \strokec4  y_train\cf6 \cb3 \strokec6 )\cf4 \cb1 \strokec4 \
\cb3 train_loader = DataLoader\cf6 \cb3 \strokec6 (\cf4 \cb3 \strokec4 train_dataset\cf6 \cb3 \strokec6 ,\cf4 \cb3 \strokec4  batch_size=\cf14 \cb3 \strokec14 32\cf6 \cb3 \strokec6 ,\cf4 \cb3 \strokec4  shuffle=\cf12 \cb3 \strokec12 True\cf6 \cb3 \strokec6 )\cf4 \cb1 \strokec4 \
\
\cb3 test_dataset = TensorDataset\cf6 \cb3 \strokec6 (\cf4 \cb3 \strokec4 x_test\cf6 \cb3 \strokec6 ,\cf4 \cb3 \strokec4  y_test\cf6 \cb3 \strokec6 )\cf4 \cb1 \strokec4 \
\cb3 test_loader = DataLoader\cf6 \cb3 \strokec6 (\cf4 \cb3 \strokec4 test_dataset\cf6 \cb3 \strokec6 ,\cf4 \cb3 \strokec4  batch_size=\cf14 \cb3 \strokec14 32\cf6 \cb3 \strokec6 ,\cf4 \cb3 \strokec4  shuffle=\cf12 \cb3 \strokec12 False\cf6 \cb3 \strokec6 )\
\
\pard\pardeftab720\partightenfactor0
\cf11 \cb3 \strokec11 # Positional Encoding for Transformer\cf4 \cb1 \strokec4 \
\pard\pardeftab720\partightenfactor0
\cf12 \cb3 \strokec12 class\cf4 \cb3 \strokec4  \cf13 \cb3 \strokec13 PositionalEncoding\cf4 \cb3 \strokec4 (\cf13 \cb3 \strokec13 pl\cf4 \cb3 \strokec4 .\cf13 \cb3 \strokec13 LightningModule\cf4 \cb3 \strokec4 )\cf6 \cb3 \strokec6 :\cf4 \cb1 \strokec4 \
\pard\pardeftab720\partightenfactor0
\cf4 \cb3     \cf12 \cb3 \strokec12 def\cf4 \cb3 \strokec4  \cf15 \cb3 \strokec15 __init__\cf4 \cb3 \strokec4 (\cf16 \cb3 \strokec16 self\cf4 \cb3 \strokec4 , \cf16 \cb3 \strokec16 d_model\cf4 \cb3 \strokec4 , \cf16 \cb3 \strokec16 dropout\cf4 \cb3 \strokec4 =\cf14 \cb3 \strokec14 0.1\cf4 \cb3 \strokec4 , \cf16 \cb3 \strokec16 max_len\cf4 \cb3 \strokec4 =\cf14 \cb3 \strokec14 500\cf4 \cb3 \strokec4 )\cf6 \cb3 \strokec6 :\cf4 \cb1 \strokec4 \
\cb3         super\cf6 \cb3 \strokec6 (\cf4 \cb3 \strokec4 PositionalEncoding\cf6 \cb3 \strokec6 ,\cf4 \cb3 \strokec4  \cf16 \cb3 \strokec16 self\cf6 \cb3 \strokec6 )\cf4 \cb3 \strokec4 .\cf15 \cb3 \strokec15 __init__\cf6 \cb3 \strokec6 ()\cf4 \cb1 \strokec4 \
\cb3         \cf16 \cb3 \strokec16 self\cf4 \cb3 \strokec4 .dropout = nn.Dropout\cf6 \cb3 \strokec6 (\cf4 \cb3 \strokec4 p=dropout\cf6 \cb3 \strokec6 )\cf4 \cb1 \strokec4 \
\
\cb3         pe = torch.zeros\cf6 \cb3 \strokec6 (\cf4 \cb3 \strokec4 max_len\cf6 \cb3 \strokec6 ,\cf4 \cb3 \strokec4  d_model\cf6 \cb3 \strokec6 )\cf4 \cb1 \strokec4 \
\cb3         position = torch.arange\cf6 \cb3 \strokec6 (\cf14 \cb3 \strokec14 0\cf6 \cb3 \strokec6 ,\cf4 \cb3 \strokec4  max_len\cf6 \cb3 \strokec6 ,\cf4 \cb3 \strokec4  dtype=torch.\cf13 \cb3 \strokec13 float\cf6 \cb3 \strokec6 )\cf4 \cb3 \strokec4 .unsqueeze\cf6 \cb3 \strokec6 (\cf14 \cb3 \strokec14 1\cf6 \cb3 \strokec6 )\cf4 \cb1 \strokec4 \
\cb3         div_term = torch.exp\cf6 \cb3 \strokec6 (\cf4 \cb3 \strokec4 torch.arange\cf6 \cb3 \strokec6 (\cf14 \cb3 \strokec14 0\cf6 \cb3 \strokec6 ,\cf4 \cb3 \strokec4  d_model\cf6 \cb3 \strokec6 ,\cf4 \cb3 \strokec4  \cf14 \cb3 \strokec14 2\cf6 \cb3 \strokec6 )\cf4 \cb3 \strokec4 .\cf13 \cb3 \strokec13 float\cf6 \cb3 \strokec6 ()\cf4 \cb3 \strokec4  * \cf6 \cb3 \strokec6 (\cf4 \cb3 \strokec4 -np.log\cf6 \cb3 \strokec6 (\cf14 \cb3 \strokec14 10000.0\cf6 \cb3 \strokec6 )\cf4 \cb3 \strokec4  / d_model\cf6 \cb3 \strokec6 ))\cf4 \cb1 \strokec4 \
\cb3         pe\cf6 \cb3 \strokec6 [:,\cf4 \cb3 \strokec4  \cf14 \cb3 \strokec14 0\cf6 \cb3 \strokec6 ::\cf14 \cb3 \strokec14 2\cf6 \cb3 \strokec6 ]\cf4 \cb3 \strokec4  = torch.sin\cf6 \cb3 \strokec6 (\cf4 \cb3 \strokec4 position * div_term\cf6 \cb3 \strokec6 )\cf4 \cb1 \strokec4 \
\cb3         pe\cf6 \cb3 \strokec6 [:,\cf4 \cb3 \strokec4  \cf14 \cb3 \strokec14 1\cf6 \cb3 \strokec6 ::\cf14 \cb3 \strokec14 2\cf6 \cb3 \strokec6 ]\cf4 \cb3 \strokec4  = torch.cos\cf6 \cb3 \strokec6 (\cf4 \cb3 \strokec4 position * div_term\cf6 \cb3 \strokec6 )\cf4 \cb1 \strokec4 \
\cb3         pe = pe.unsqueeze\cf6 \cb3 \strokec6 (\cf14 \cb3 \strokec14 0\cf6 \cb3 \strokec6 )\cf4 \cb3 \strokec4 .transpose\cf6 \cb3 \strokec6 (\cf14 \cb3 \strokec14 0\cf6 \cb3 \strokec6 ,\cf4 \cb3 \strokec4  \cf14 \cb3 \strokec14 1\cf6 \cb3 \strokec6 )\cf4 \cb1 \strokec4 \
\cb3         \cf16 \cb3 \strokec16 self\cf4 \cb3 \strokec4 .register_buffer\cf6 \cb3 \strokec6 (\cf10 \strokec10 'pe'\cf6 \strokec6 ,\cf4 \cb3 \strokec4  pe\cf6 \cb3 \strokec6 )\cf4 \cb1 \strokec4 \
\
\cb3     \cf12 \cb3 \strokec12 def\cf4 \cb3 \strokec4  \cf15 \cb3 \strokec15 forward\cf4 \cb3 \strokec4 (\cf16 \cb3 \strokec16 self\cf4 \cb3 \strokec4 , \cf16 \cb3 \strokec16 x\cf4 \cb3 \strokec4 )\cf6 \cb3 \strokec6 :\cf4 \cb1 \strokec4 \
\cb3         x = x + \cf16 \cb3 \strokec16 self\cf4 \cb3 \strokec4 .pe\cf6 \cb3 \strokec6 [:\cf4 \cb3 \strokec4 x.size\cf6 \cb3 \strokec6 (\cf14 \cb3 \strokec14 0\cf6 \cb3 \strokec6 ),\cf4 \cb3 \strokec4  \cf6 \cb3 \strokec6 :]\cf4 \cb1 \strokec4 \
\cb3         \cf5 \cb3 \strokec5 return\cf4 \cb3 \strokec4  \cf16 \cb3 \strokec16 self\cf4 \cb3 \strokec4 .dropout\cf6 \cb3 \strokec6 (\cf4 \cb3 \strokec4 x\cf6 \cb3 \strokec6 )\cf4 \cb1 \strokec4 \
\
\cb3 device = \cf6 \cb3 \strokec6 (\cf4 \cb1 \strokec4 \
\cb3     \cf10 \cb3 \strokec10 "mps"\cf4 \cb1 \strokec4 \
\cb3     \cf5 \cb3 \strokec5 if\cf4 \cb3 \strokec4  \cf15 \cb3 \strokec15 getattr\cf6 \cb3 \strokec6 (\cf4 \cb3 \strokec4 torch\cf6 \cb3 \strokec6 ,\cf4 \cb3 \strokec4  \cf10 \cb3 \strokec10 "has_mps"\cf6 \strokec6 ,\cf4 \cb3 \strokec4  \cf12 \cb3 \strokec12 False\cf6 \cb3 \strokec6 )\cf4 \cb1 \strokec4 \
\cb3     \cf5 \cb3 \strokec5 else\cf4 \cb3 \strokec4  \cf10 \cb3 \strokec10 "cuda"\cf4 \cb1 \strokec4 \
\cb3     \cf5 \cb3 \strokec5 if\cf4 \cb3 \strokec4  torch.cuda.is_available\cf6 \cb3 \strokec6 ()\cf4 \cb1 \strokec4 \
\cb3     \cf5 \cb3 \strokec5 else\cf4 \cb3 \strokec4  \cf10 \cb3 \strokec10 "cpu"\cf4 \cb1 \strokec4 \
\pard\pardeftab720\partightenfactor0
\cf6 \cb3 \strokec6 )\cf4 \cb1 \strokec4 \
\
#Transformer model\
\pard\pardeftab720\partightenfactor0
\cf12 \cb3 \strokec12 class\cf4 \cb3 \strokec4  \cf13 \cb3 \strokec13 TimeSeriesTransformer\cf4 \cb3 \strokec4 (\cf13 \cb3 \strokec13 pl\cf4 \cb3 \strokec4 .\cf13 \cb3 \strokec13 LightningModule\cf4 \cb3 \strokec4 )\cf6 \cb3 \strokec6 :\cf4 \cb1 \strokec4 \
\pard\pardeftab720\partightenfactor0
\cf4 \cb3     \cf12 \cb3 \strokec12 def\cf4 \cb3 \strokec4  \cf15 \cb3 \strokec15 __init__\cf4 \cb3 \strokec4 (\cf16 \cb3 \strokec16 self\cf4 \cb3 \strokec4 , \cf16 \cb3 \strokec16 input_dim\cf4 \cb3 \strokec4 =\cf14 \cb3 \strokec14 2\cf4 \cb3 \strokec4 , \cf16 \cb3 \strokec16 d_model\cf4 \cb3 \strokec4 =\cf14 \cb3 \strokec14 512\cf4 \cb3 \strokec4 , \cf16 \cb3 \strokec16 nhead\cf4 \cb3 \strokec4 =\cf14 \cb3 \strokec14 4\cf4 \cb3 \strokec4 , \cf16 \cb3 \strokec16 num_layers\cf4 \cb3 \strokec4 =\cf14 \cb3 \strokec14 2\cf4 \cb3 \strokec4 , \cf16 \cb3 \strokec16 dropout\cf4 \cb3 \strokec4 =\cf14 \cb3 \strokec14 0.2\cf4 \cb3 \strokec4 )\cf6 \cb3 \strokec6 :\cf4 \cb1 \strokec4 \
\cb3         super\cf6 \cb3 \strokec6 ()\cf4 \cb3 \strokec4 .\cf15 \cb3 \strokec15 __init__\cf6 \cb3 \strokec6 ()\cf4 \cb1 \strokec4 \
\
\cb3         \cf16 \cb3 \strokec16 self\cf4 \cb3 \strokec4 .encoder = nn.Linear\cf6 \cb3 \strokec6 (\cf4 \cb3 \strokec4 input_dim\cf6 \cb3 \strokec6 ,\cf4 \cb3 \strokec4  d_model\cf6 \cb3 \strokec6 )\cf4 \cb1 \strokec4 \
\cb3         \cf11 \cb3 \strokec11 # self.pos_encoder = PositionalEncoding(d_model, dropout)\cf4 \cb1 \strokec4 \
\cb3         \cf16 \cb3 \strokec16 self\cf4 \cb3 \strokec4 .pos_encoder = nn.Embedding\cf6 \cb3 \strokec6 (\cf15 \cb3 \strokec15 len\cf6 \cb3 \strokec6 (\cf4 \cb3 \strokec4 df_train\cf6 \cb3 \strokec6 ),\cf4 \cb3 \strokec4  d_model\cf6 \cb3 \strokec6 )\cf4 \cb1 \strokec4 \
\cb3         encoder_layers = nn.TransformerEncoderLayer\cf6 \cb3 \strokec6 (\cf4 \cb3 \strokec4 d_model\cf6 \cb3 \strokec6 ,\cf4 \cb3 \strokec4  nhead\cf6 \cb3 \strokec6 )\cf4 \cb1 \strokec4 \
\cb3         \cf16 \cb3 \strokec16 self\cf4 \cb3 \strokec4 .transformer_encoder = nn.TransformerEncoder\cf6 \cb3 \strokec6 (\cf4 \cb3 \strokec4 encoder_layers\cf6 \cb3 \strokec6 ,\cf4 \cb3 \strokec4  num_layers\cf6 \cb3 \strokec6 )\cf4 \cb1 \strokec4 \
\cb3         \cf16 \cb3 \strokec16 self\cf4 \cb3 \strokec4 .decoder = nn.Linear\cf6 \cb3 \strokec6 (\cf4 \cb3 \strokec4 d_model\cf6 \cb3 \strokec6 ,\cf4 \cb3 \strokec4  \cf14 \cb3 \strokec14 2\cf6 \cb3 \strokec6 )\cf4 \cb1 \strokec4 \
\
\cb3         \cf16 \cb3 \strokec16 self\cf4 \cb3 \strokec4 .criterion = nn.MSELoss\cf6 \cb3 \strokec6 ()\cf4 \cb1 \strokec4 \
\cb3         \cf16 \cb3 \strokec16 self\cf4 \cb3 \strokec4 .learning_rate = \cf14 \cb3 \strokec14 0.01\cf4 \cb1 \strokec4 \
\
\cb3     \cf12 \cb3 \strokec12 def\cf4 \cb3 \strokec4  \cf15 \cb3 \strokec15 forward\cf4 \cb3 \strokec4 (\cf16 \cb3 \strokec16 self\cf4 \cb3 \strokec4 , \cf16 \cb3 \strokec16 x\cf4 \cb3 \strokec4 )\cf6 \cb3 \strokec6 :\cf4 \cb1 \strokec4 \
\cb3         x = \cf16 \cb3 \strokec16 self\cf4 \cb3 \strokec4 .encoder\cf6 \cb3 \strokec6 (\cf4 \cb3 \strokec4 x\cf6 \cb3 \strokec6 )\cf4 \cb1 \strokec4 \
\cb3         l = torch.arange\cf6 \cb3 \strokec6 (\cf15 \cb3 \strokec15 len\cf6 \cb3 \strokec6 (\cf4 \cb3 \strokec4 x\cf6 \cb3 \strokec6 [\cf14 \cb3 \strokec14 1\cf6 \cb3 \strokec6 ]))\cf4 \cb1 \strokec4 \
\cb3         l = l.to\cf6 \cb3 \strokec6 (\cf4 \cb3 \strokec4 device\cf6 \cb3 \strokec6 )\cf4 \cb1 \strokec4 \
\
\
\
\cb3         x = \cf6 \cb3 \strokec6 [\cf4 \cb3 \strokec4 x\cf6 \cb3 \strokec6 [\cf4 \cb3 \strokec4 i\cf6 \cb3 \strokec6 ]\cf4 \cb3 \strokec4 +\cf16 \cb3 \strokec16 self\cf4 \cb3 \strokec4 .pos_encoder\cf6 \cb3 \strokec6 (\cf4 \cb3 \strokec4 l\cf6 \cb3 \strokec6 )\cf4 \cb3 \strokec4  \cf5 \cb3 \strokec5 for\cf4 \cb3 \strokec4  i \cf2 \strokec2 in\cf4 \strokec4  \cf15 \cb3 \strokec15 range\cf6 \cb3 \strokec6 (\cf4 \cb3 \strokec4 x.size\cf6 \cb3 \strokec6 ()[\cf14 \cb3 \strokec14 0\cf6 \cb3 \strokec6 ])]\cf4 \cb1 \strokec4 \
\cb3         x=torch.stack\cf6 \cb3 \strokec6 (\cf4 \cb3 \strokec4 x\cf6 \cb3 \strokec6 )\cf4 \cb1 \strokec4 \
\
\
\cb3         x = \cf16 \cb3 \strokec16 self\cf4 \cb3 \strokec4 .transformer_encoder\cf6 \cb3 \strokec6 (\cf4 \cb3 \strokec4 x\cf6 \cb3 \strokec6 )\cf4 \cb1 \strokec4 \
\cb3         x = \cf16 \cb3 \strokec16 self\cf4 \cb3 \strokec4 .decoder\cf6 \cb3 \strokec6 (\cf4 \cb3 \strokec4 x\cf6 \cb3 \strokec6 [:,\cf4 \cb3 \strokec4  \cf14 \cb3 \strokec14 -1\cf6 \cb3 \strokec6 ,\cf4 \cb3 \strokec4  \cf6 \cb3 \strokec6 :])\cf4 \cb1 \strokec4 \
\cb3         \cf5 \cb3 \strokec5 return\cf4 \cb3 \strokec4  x\cb1 \
\
\cb3     \cf12 \cb3 \strokec12 def\cf4 \cb3 \strokec4  \cf15 \cb3 \strokec15 configure_optimizers\cf4 \cb3 \strokec4 (\cf16 \cb3 \strokec16 self\cf4 \cb3 \strokec4 )\cf6 \cb3 \strokec6 :\cf4 \cb1 \strokec4 \
\cb3         optimizer = torch.optim.AdamW\cf6 \cb3 \strokec6 (\cf16 \cb3 \strokec16 self\cf4 \cb3 \strokec4 .parameters\cf6 \cb3 \strokec6 ())\cf4 \cb1 \strokec4 \
\cb3         scheduler = ReduceLROnPlateau\cf6 \cb3 \strokec6 (\cf4 \cb3 \strokec4 optimizer\cf6 \cb3 \strokec6 ,\cf4 \cb3 \strokec4  \cf10 \cb3 \strokec10 'min'\cf6 \strokec6 ,\cf4 \cb3 \strokec4  factor=\cf14 \cb3 \strokec14 0.5\cf6 \cb3 \strokec6 ,\cf4 \cb3 \strokec4  patience=\cf14 \cb3 \strokec14 3\cf6 \cb3 \strokec6 ,\cf4 \cb3 \strokec4  verbose=\cf12 \cb3 \strokec12 True\cf6 \cb3 \strokec6 )\cf4 \cb1 \strokec4 \
\cb3         \cf5 \cb3 \strokec5 return\cf4 \cb3 \strokec4  \cf6 \cb3 \strokec6 \{\cf10 \strokec10 'optimizer'\cf6 \strokec6 :\cf4 \cb3 \strokec4  optimizer\cf6 \cb3 \strokec6 ,\cf4 \cb3 \strokec4  \cf10 \cb3 \strokec10 'lr_scheduler'\cf6 \strokec6 :\cf4 \cb3 \strokec4  scheduler\cf6 \cb3 \strokec6 ,\cf4 \cb3 \strokec4  \cf10 \cb3 \strokec10 'monitor'\cf6 \strokec6 :\cf4 \cb3 \strokec4  \cf10 \cb3 \strokec10 'val_loss'\cf6 \strokec6 \}\cf4 \cb1 \strokec4 \
\
\cb3     \cf12 \cb3 \strokec12 def\cf4 \cb3 \strokec4  \cf15 \cb3 \strokec15 training_step\cf4 \cb3 \strokec4 (\cf16 \cb3 \strokec16 self\cf4 \cb3 \strokec4 , \cf16 \cb3 \strokec16 batch\cf4 \cb3 \strokec4 , \cf16 \cb3 \strokec16 batch_idx\cf4 \cb3 \strokec4 )\cf6 \cb3 \strokec6 :\cf4 \cb1 \strokec4 \
\cb3         x_batch\cf6 \cb3 \strokec6 ,\cf4 \cb3 \strokec4  y_batch = batch\cb1 \
\cb3         outputs = \cf16 \cb3 \strokec16 self\cf6 \cb3 \strokec6 (\cf4 \cb3 \strokec4 x_batch\cf6 \cb3 \strokec6 )\cf4 \cb1 \strokec4 \
\cb3         loss = \cf16 \cb3 \strokec16 self\cf4 \cb3 \strokec4 .criterion\cf6 \cb3 \strokec6 (\cf4 \cb3 \strokec4 outputs\cf6 \cb3 \strokec6 ,\cf4 \cb3 \strokec4  y_batch\cf6 \cb3 \strokec6 )\cf4 \cb1 \strokec4 \
\cb3         \cf16 \cb3 \strokec16 self\cf4 \cb3 \strokec4 .log\cf6 \cb3 \strokec6 (\cf10 \strokec10 "train_loss"\cf6 \strokec6 ,\cf4 \cb3 \strokec4  loss\cf6 \cb3 \strokec6 ,\cf4 \cb3 \strokec4  on_step=\cf12 \cb3 \strokec12 True\cf6 \cb3 \strokec6 ,\cf4 \cb3 \strokec4  on_epoch=\cf12 \cb3 \strokec12 True\cf6 \cb3 \strokec6 ,\cf4 \cb3 \strokec4  prog_bar=\cf12 \cb3 \strokec12 True\cf6 \cb3 \strokec6 )\cf4 \cb1 \strokec4 \
\cb3         \cf5 \cb3 \strokec5 return\cf4 \cb3 \strokec4  loss\cb1 \
\
\cb3     \cf12 \cb3 \strokec12 def\cf4 \cb3 \strokec4  \cf15 \cb3 \strokec15 validation_step\cf4 \cb3 \strokec4 (\cf16 \cb3 \strokec16 self\cf4 \cb3 \strokec4 , \cf16 \cb3 \strokec16 batch\cf4 \cb3 \strokec4 , \cf16 \cb3 \strokec16 batch_idx\cf4 \cb3 \strokec4 )\cf6 \cb3 \strokec6 :\cf4 \cb1 \strokec4 \
\cb3         x_batch\cf6 \cb3 \strokec6 ,\cf4 \cb3 \strokec4  y_batch = batch\cb1 \
\cb3         outputs = \cf16 \cb3 \strokec16 self\cf6 \cb3 \strokec6 (\cf4 \cb3 \strokec4 x_batch\cf6 \cb3 \strokec6 )\cf4 \cb1 \strokec4 \
\cb3         loss = \cf16 \cb3 \strokec16 self\cf4 \cb3 \strokec4 .criterion\cf6 \cb3 \strokec6 (\cf4 \cb3 \strokec4 outputs\cf6 \cb3 \strokec6 ,\cf4 \cb3 \strokec4  y_batch\cf6 \cb3 \strokec6 )\cf4 \cb1 \strokec4 \
\cb3         \cf16 \cb3 \strokec16 self\cf4 \cb3 \strokec4 .log\cf6 \cb3 \strokec6 (\cf10 \strokec10 "val_loss"\cf6 \strokec6 ,\cf4 \cb3 \strokec4  loss\cf6 \cb3 \strokec6 ,\cf4 \cb3 \strokec4  on_step=\cf12 \cb3 \strokec12 False\cf6 \cb3 \strokec6 ,\cf4 \cb3 \strokec4  on_epoch=\cf12 \cb3 \strokec12 True\cf6 \cb3 \strokec6 ,\cf4 \cb3 \strokec4  prog_bar=\cf12 \cb3 \strokec12 True\cf6 \cb3 \strokec6 )\cf4 \cb1 \strokec4 \
\cb3         \cf5 \cb3 \strokec5 return\cf4 \cb3 \strokec4  loss\cb1 \
\
\
\cb3 model = TimeSeriesTransformer\cf6 \cb3 \strokec6 ()\cf4 \cb1 \strokec4 \
\
\
\cb3 shutil.rmtree\cf6 \cb3 \strokec6 (\cf10 \strokec10 '/content/lightning_logs'\cf6 \strokec6 )\
\
\pard\pardeftab720\partightenfactor0
\cf11 \cb3 \strokec11 # Setup model checkpoint callback\cf4 \cb1 \strokec4 \
\pard\pardeftab720\partightenfactor0
\cf4 \cb3 checkpoint_callback = pl.callbacks.ModelCheckpoint\cf6 \cb3 \strokec6 (\cf4 \cb1 \strokec4 \
\cb3     dirpath=\cf10 \cb3 \strokec10 "lightning_logs"\cf6 \strokec6 ,\cf4 \cb1 \strokec4 \
\cb3     filename=\cf10 \cb3 \strokec10 "best-checkpoint"\cf6 \strokec6 ,\cf4 \cb1 \strokec4 \
\cb3     save_top_k = \cf14 \cb3 \strokec14 1\cf6 \cb3 \strokec6 ,\cf4 \cb1 \strokec4 \
\cb3     verbose = \cf12 \cb3 \strokec12 True\cf6 \cb3 \strokec6 ,\cf4 \cb1 \strokec4 \
\cb3     monitor = \cf10 \cb3 \strokec10 "val_loss"\cf6 \strokec6 ,\cf4 \cb1 \strokec4 \
\cb3     mode = \cf10 \cb3 \strokec10 "min"\cf4 \cb1 \strokec4 \
\cb3     \cf6 \cb3 \strokec6 )\cf4 \cb1 \strokec4 \
\
\cb3 logger = TensorBoardLogger\cf6 \cb3 \strokec6 (\cf10 \strokec10 "lightning_logs"\cf6 \strokec6 ,\cf4 \cb3 \strokec4  name = \cf10 \cb3 \strokec10 "ts_model"\cf6 \strokec6 )\cf4 \cb1 \strokec4 \
\cb3 early_stopping_callback = EarlyStopping\cf6 \cb3 \strokec6 (\cf4 \cb3 \strokec4 monitor = \cf10 \cb3 \strokec10 "val_loss"\cf6 \strokec6 ,\cf4 \cb3 \strokec4  patience = \cf14 \cb3 \strokec14 2\cf6 \cb3 \strokec6 )\cf4 \cb1 \strokec4 \
\
\pard\pardeftab720\partightenfactor0
\cf11 \cb3 \strokec11 # Setup PyTorch Lightning Trainer\cf4 \cb1 \strokec4 \
\pard\pardeftab720\partightenfactor0
\cf4 \cb3 trainer = pl.Trainer\cf6 \cb3 \strokec6 (\cf4 \cb1 \strokec4 \
\cb3     max_epochs=\cf14 \cb3 \strokec14 5\cf6 \cb3 \strokec6 ,\cf4 \cb1 \strokec4 \
\cb3     logger = logger\cf6 \cb3 \strokec6 ,\cf4 \cb1 \strokec4 \
\cb3     callbacks=\cf6 \cb3 \strokec6 [\cf4 \cb3 \strokec4 early_stopping_callback\cf6 \cb3 \strokec6 ],\cf4 \cb1 \strokec4 \
\cb3     \cf11 \cb3 \strokec11 #gpus=1 if torch.cuda.is_available() else None\cf4 \cb1 \strokec4 \
\cb3     \cf6 \cb3 \strokec6 )\cf4 \cb1 \strokec4 \
\
\cb3 trainer.fit\cf6 \cb3 \strokec6 (\cf4 \cb3 \strokec4 model\cf6 \cb3 \strokec6 ,\cf4 \cb3 \strokec4  train_loader\cf6 \cb3 \strokec6 ,\cf4 \cb3 \strokec4  test_loader\cf6 \cb3 \strokec6 )\cf4 \cb1 \strokec4 \
\
\pard\pardeftab720\partightenfactor0
\cf11 \cb3 \strokec11 # Evaluation\cf4 \cb1 \strokec4 \
\pard\pardeftab720\partightenfactor0
\cf4 \cb3 checkpoint_path=\cf10 \cb3 \strokec10 \'93Add path to .ckpt here\'94\cf4 \cb1 \strokec4 \
\cb3 model = TimeSeriesTransformer.load_from_checkpoint\cf6 \cb3 \strokec6 (\cf4 \cb3 \strokec4 checkpoint_path\cf6 \cb3 \strokec6 )\cf4 \cb1 \strokec4 \
\
\cb3 model.\cf15 \cb3 \strokec15 eval\cf6 \cb3 \strokec6 ()\cf4 \cb1 \strokec4 \
\cb3 predictions = \cf6 \cb3 \strokec6 []\cf4 \cb1 \strokec4 \
\pard\pardeftab720\partightenfactor0
\cf5 \cb3 \strokec5 with\cf4 \cb3 \strokec4  torch.no_grad\cf6 \cb3 \strokec6 ():\cf4 \cb1 \strokec4 \
\pard\pardeftab720\partightenfactor0
\cf4 \cb3     \cf5 \cb3 \strokec5 for\cf4 \cb3 \strokec4  batch \cf2 \strokec2 in\cf4 \strokec4  test_loader\cf6 \cb3 \strokec6 :\cf4 \cb1 \strokec4 \
\cb3         x_batch\cf6 \cb3 \strokec6 ,\cf4 \cb3 \strokec4  y_batch = batch\cb1 \
\cb3         x_batch = x_batch.to\cf6 \cb3 \strokec6 (\cf4 \cb3 \strokec4 device\cf6 \cb3 \strokec6 )\cf4 \cb1 \strokec4 \
\cb3         outputs = model\cf6 \cb3 \strokec6 (\cf4 \cb3 \strokec4 x_batch\cf6 \cb3 \strokec6 )\cf4 \cb1 \strokec4 \
\cb3         predictions.extend\cf6 \cb3 \strokec6 (\cf4 \cb3 \strokec4 outputs.tolist\cf6 \cb3 \strokec6 ())\cf4 \cb1 \strokec4 \
\
\pard\pardeftab720\partightenfactor0
\cf11 \cb3 \strokec11 # Convert predictions and labels back to their original scale\cf4 \cb1 \strokec4 \
\pard\pardeftab720\partightenfactor0
\cf4 \cb3 predictions = scaler.inverse_transform\cf6 \cb3 \strokec6 (\cf4 \cb3 \strokec4 np.array\cf6 \cb3 \strokec6 (\cf4 \cb3 \strokec4 predictions\cf6 \cb3 \strokec6 ))\cf4 \cb1 \strokec4 \
\cb3 labels = scaler.inverse_transform\cf6 \cb3 \strokec6 (\cf4 \cb3 \strokec4 y_test.numpy\cf6 \cb3 \strokec6 ())\cf4 \cb1 \strokec4 \
\
\pard\pardeftab720\partightenfactor0
\cf11 \cb3 \strokec11 # Calculate RMSE considering 2D structure\cf4 \cb1 \strokec4 \
\pard\pardeftab720\partightenfactor0
\cf4 \cb3 rmse = np.sqrt\cf6 \cb3 \strokec6 (\cf4 \cb3 \strokec4 np.mean\cf6 \cb3 \strokec6 ((\cf4 \cb3 \strokec4 predictions - labels\cf6 \cb3 \strokec6 )\cf4 \cb3 \strokec4  ** \cf14 \cb3 \strokec14 2\cf6 \cb3 \strokec6 ))\cf4 \cb1 \strokec4 \
\pard\pardeftab720\partightenfactor0
\cf15 \cb3 \strokec15 print\cf6 \cb3 \strokec6 (\cf12 \cb3 \strokec12 f\cf10 \cb3 \strokec10 "Score (RMSE): \cf6 \strokec6 \{\cf4 \cb3 \strokec4 rmse\cf14 \cb3 \strokec14 :.4f\cf6 \cb3 \strokec6 \}\cf10 \strokec10 "\cf6 \strokec6 )\
\
\pard\pardeftab720\partightenfactor0
\cf4 \cb3 \strokec4 unique\cf6 \cb3 \strokec6 ,\cf4 \cb3 \strokec4  counts = np.unique\cf6 \cb3 \strokec6 (\cf4 \cb3 \strokec4 labels_2\cf6 \cb3 \strokec6 ,\cf4 \cb3 \strokec4  return_counts=\cf12 \cb3 \strokec12 True\cf6 \cb3 \strokec6 )\cf4 \cb1 \strokec4 \
\pard\pardeftab720\partightenfactor0
\cf13 \cb3 \strokec13 dict\cf6 \cb3 \strokec6 (\cf15 \cb3 \strokec15 zip\cf6 \cb3 \strokec6 (\cf4 \cb3 \strokec4 unique\cf6 \cb3 \strokec6 ,\cf4 \cb3 \strokec4  counts\cf6 \cb3 \strokec6 ))\cf4 \cb1 \strokec4 \
\
\pard\pardeftab720\partightenfactor0
\cf4 \cb3 temp=predictions_2\cb1 \
\cb3 x=temp.size\cb1 \
\
\pard\pardeftab720\partightenfactor0
\cf5 \cb3 \strokec5 for\cf4 \cb3 \strokec4  j \cf2 \strokec2 in\cf4 \strokec4  \cf15 \cb3 \strokec15 range\cf6 \cb3 \strokec6 (\cf14 \cb3 \strokec14 0\cf6 \cb3 \strokec6 ,\cf4 \cb3 \strokec4 x\cf6 \cb3 \strokec6 ):\cf4 \cb1 \strokec4 \
\pard\pardeftab720\partightenfactor0
\cf4 \cb3     \cf5 \cb3 \strokec5 if\cf4 \cb3 \strokec4  temp\cf6 \cb3 \strokec6 [\cf4 \cb3 \strokec4 j\cf6 \cb3 \strokec6 ]\cf4 \cb3 \strokec4 <\cf14 \cb3 \strokec14 0.33219855\cf6 \cb3 \strokec6 :\cf4 \cb1 \strokec4 \
\cb3       temp\cf6 \cb3 \strokec6 [\cf4 \cb3 \strokec4 j\cf6 \cb3 \strokec6 ]\cf4 \cb3 \strokec4 =\cf14 \cb3 \strokec14 0\cf4 \cb1 \strokec4 \
\cb3     \cf5 \cb3 \strokec5 elif\cf4 \cb3 \strokec4  temp\cf6 \cb3 \strokec6 [\cf4 \cb3 \strokec4 j\cf6 \cb3 \strokec6 ]\cf4 \cb3 \strokec4 >=\cf14 \cb3 \strokec14 0.33219855\cf4 \cb3 \strokec4  \cf2 \strokec2 and\cf4 \strokec4  temp\cf6 \cb3 \strokec6 [\cf4 \cb3 \strokec4 j\cf6 \cb3 \strokec6 ]\cf4 \cb3 \strokec4 <\cf14 \cb3 \strokec14 0.33219856\cf6 \cb3 \strokec6 :\cf4 \cb1 \strokec4 \
\cb3       temp\cf6 \cb3 \strokec6 [\cf4 \cb3 \strokec4 j\cf6 \cb3 \strokec6 ]\cf4 \cb3 \strokec4 =\cf14 \cb3 \strokec14 4\cf4 \cb1 \strokec4 \
\cb3     \cf5 \cb3 \strokec5 elif\cf4 \cb3 \strokec4  temp\cf6 \cb3 \strokec6 [\cf4 \cb3 \strokec4 j\cf6 \cb3 \strokec6 ]\cf4 \cb3 \strokec4 >=\cf14 \cb3 \strokec14 0.33219856\cf4 \cb3 \strokec4  \cf2 \strokec2 and\cf4 \strokec4  temp\cf6 \cb3 \strokec6 [\cf4 \cb3 \strokec4 j\cf6 \cb3 \strokec6 ]\cf4 \cb3 \strokec4 <\cf14 \cb3 \strokec14 0.3321986\cf6 \cb3 \strokec6 :\cf4 \cb1 \strokec4 \
\cb3       temp\cf6 \cb3 \strokec6 [\cf4 \cb3 \strokec4 j\cf6 \cb3 \strokec6 ]\cf4 \cb3 \strokec4 =\cf14 \cb3 \strokec14 5\cf4 \cb1 \strokec4 \
\cb3     \cf5 \cb3 \strokec5 elif\cf4 \cb3 \strokec4  temp\cf6 \cb3 \strokec6 [\cf4 \cb3 \strokec4 j\cf6 \cb3 \strokec6 ]\cf4 \cb3 \strokec4 >=\cf14 \cb3 \strokec14 0.3321986\cf4 \cb3 \strokec4  \cf2 \strokec2 and\cf4 \strokec4  temp\cf6 \cb3 \strokec6 [\cf4 \cb3 \strokec4 j\cf6 \cb3 \strokec6 ]\cf4 \cb3 \strokec4 <\cf14 \cb3 \strokec14 0.3321987\cf6 \cb3 \strokec6 :\cf4 \cb1 \strokec4 \
\cb3       temp\cf6 \cb3 \strokec6 [\cf4 \cb3 \strokec4 j\cf6 \cb3 \strokec6 ]\cf4 \cb3 \strokec4 =\cf14 \cb3 \strokec14 6\cf4 \cb1 \strokec4 \
\cb3     \cf5 \cb3 \strokec5 elif\cf4 \cb3 \strokec4  temp\cf6 \cb3 \strokec6 [\cf4 \cb3 \strokec4 j\cf6 \cb3 \strokec6 ]\cf4 \cb3 \strokec4 >=\cf14 \cb3 \strokec14 0.3321987\cf4 \cb3 \strokec4  \cf2 \strokec2 and\cf4 \strokec4  temp\cf6 \cb3 \strokec6 [\cf4 \cb3 \strokec4 j\cf6 \cb3 \strokec6 ]\cf4 \cb3 \strokec4 <\cf14 \cb3 \strokec14 0.332199\cf6 \cb3 \strokec6 :\cf4 \cb1 \strokec4 \
\cb3       temp\cf6 \cb3 \strokec6 [\cf4 \cb3 \strokec4 j\cf6 \cb3 \strokec6 ]\cf4 \cb3 \strokec4 =\cf14 \cb3 \strokec14 8\cf4 \cb1 \strokec4 \
\cb3     \cf5 \cb3 \strokec5 elif\cf4 \cb3 \strokec4  temp\cf6 \cb3 \strokec6 [\cf4 \cb3 \strokec4 j\cf6 \cb3 \strokec6 ]\cf4 \cb3 \strokec4 >=\cf14 \cb3 \strokec14 0.332199\cf6 \cb3 \strokec6 :\cf4 \cb1 \strokec4 \
\cb3       temp\cf6 \cb3 \strokec6 [\cf4 \cb3 \strokec4 j\cf6 \cb3 \strokec6 ]\cf4 \cb3 \strokec4 =\cf14 \cb3 \strokec14 9\cf4 \cb1 \strokec4 \
\
\cb3 unique\cf6 \cb3 \strokec6 ,\cf4 \cb3 \strokec4  counts = np.unique\cf6 \cb3 \strokec6 (\cf4 \cb3 \strokec4 temp\cf6 \cb3 \strokec6 ,\cf4 \cb3 \strokec4  return_counts=\cf12 \cb3 \strokec12 True\cf6 \cb3 \strokec6 )\cf4 \cb1 \strokec4 \
\pard\pardeftab720\partightenfactor0
\cf13 \cb3 \strokec13 dict\cf6 \cb3 \strokec6 (\cf15 \cb3 \strokec15 zip\cf6 \cb3 \strokec6 (\cf4 \cb3 \strokec4 unique\cf6 \cb3 \strokec6 ,\cf4 \cb3 \strokec4  counts\cf6 \cb3 \strokec6 ))\cf4 \cb1 \strokec4 \
\
\pard\pardeftab720\partightenfactor0
\cf4 \cb3 actual=labels_2\cb1 \
\cb3 actual_size=actual.size\cb1 \
\
\pard\pardeftab720\partightenfactor0
\cf5 \cb3 \strokec5 for\cf4 \cb3 \strokec4  j \cf2 \strokec2 in\cf4 \strokec4  \cf15 \cb3 \strokec15 range\cf6 \cb3 \strokec6 (\cf14 \cb3 \strokec14 0\cf6 \cb3 \strokec6 ,\cf4 \cb3 \strokec4 actual_size\cf6 \cb3 \strokec6 ):\cf4 \cb1 \strokec4 \
\pard\pardeftab720\partightenfactor0
\cf4 \cb3     \cf5 \cb3 \strokec5 if\cf4 \cb3 \strokec4  actual\cf6 \cb3 \strokec6 [\cf4 \cb3 \strokec4 j\cf6 \cb3 \strokec6 ]\cf4 \cb3 \strokec4 <\cf14 \cb3 \strokec14 2.25\cf6 \cb3 \strokec6 :\cf4 \cb1 \strokec4 \
\cb3       actual\cf6 \cb3 \strokec6 [\cf4 \cb3 \strokec4 j\cf6 \cb3 \strokec6 ]\cf4 \cb3 \strokec4 =\cf14 \cb3 \strokec14 0\cf4 \cb1 \strokec4 \
\cb3     \cf5 \cb3 \strokec5 elif\cf4 \cb3 \strokec4  actual\cf6 \cb3 \strokec6 [\cf4 \cb3 \strokec4 j\cf6 \cb3 \strokec6 ]\cf4 \cb3 \strokec4 >=\cf14 \cb3 \strokec14 2.25\cf4 \cb3 \strokec4  \cf2 \strokec2 and\cf4 \strokec4  actual\cf6 \cb3 \strokec6 [\cf4 \cb3 \strokec4 j\cf6 \cb3 \strokec6 ]\cf4 \cb3 \strokec4 <\cf14 \cb3 \strokec14 5.0\cf6 \cb3 \strokec6 :\cf4 \cb1 \strokec4 \
\cb3       actual\cf6 \cb3 \strokec6 [\cf4 \cb3 \strokec4 j\cf6 \cb3 \strokec6 ]\cf4 \cb3 \strokec4 =\cf14 \cb3 \strokec14 4\cf4 \cb1 \strokec4 \
\cb3     \cf5 \cb3 \strokec5 elif\cf4 \cb3 \strokec4  actual\cf6 \cb3 \strokec6 [\cf4 \cb3 \strokec4 j\cf6 \cb3 \strokec6 ]\cf4 \cb3 \strokec4 >=\cf14 \cb3 \strokec14 5.0\cf4 \cb3 \strokec4  \cf2 \strokec2 and\cf4 \strokec4  actual\cf6 \cb3 \strokec6 [\cf4 \cb3 \strokec4 j\cf6 \cb3 \strokec6 ]\cf4 \cb3 \strokec4 <\cf14 \cb3 \strokec14 5.875\cf6 \cb3 \strokec6 :\cf4 \cb1 \strokec4 \
\cb3       actual\cf6 \cb3 \strokec6 [\cf4 \cb3 \strokec4 j\cf6 \cb3 \strokec6 ]\cf4 \cb3 \strokec4 =\cf14 \cb3 \strokec14 5\cf4 \cb1 \strokec4 \
\cb3     \cf5 \cb3 \strokec5 elif\cf4 \cb3 \strokec4  actual\cf6 \cb3 \strokec6 [\cf4 \cb3 \strokec4 j\cf6 \cb3 \strokec6 ]\cf4 \cb3 \strokec4 >=\cf14 \cb3 \strokec14 5.875\cf4 \cb3 \strokec4  \cf2 \strokec2 and\cf4 \strokec4  actual\cf6 \cb3 \strokec6 [\cf4 \cb3 \strokec4 j\cf6 \cb3 \strokec6 ]\cf4 \cb3 \strokec4 <\cf14 \cb3 \strokec14 7.125\cf6 \cb3 \strokec6 :\cf4 \cb1 \strokec4 \
\cb3       actual\cf6 \cb3 \strokec6 [\cf4 \cb3 \strokec4 j\cf6 \cb3 \strokec6 ]\cf4 \cb3 \strokec4 =\cf14 \cb3 \strokec14 6\cf4 \cb1 \strokec4 \
\cb3     \cf5 \cb3 \strokec5 elif\cf4 \cb3 \strokec4  actual\cf6 \cb3 \strokec6 [\cf4 \cb3 \strokec4 j\cf6 \cb3 \strokec6 ]\cf4 \cb3 \strokec4 >=\cf14 \cb3 \strokec14 7.125\cf4 \cb3 \strokec4  \cf2 \strokec2 and\cf4 \strokec4  actual\cf6 \cb3 \strokec6 [\cf4 \cb3 \strokec4 j\cf6 \cb3 \strokec6 ]\cf4 \cb3 \strokec4 <\cf14 \cb3 \strokec14 8.5\cf6 \cb3 \strokec6 :\cf4 \cb1 \strokec4 \
\cb3       actual\cf6 \cb3 \strokec6 [\cf4 \cb3 \strokec4 j\cf6 \cb3 \strokec6 ]\cf4 \cb3 \strokec4 =\cf14 \cb3 \strokec14 8\cf4 \cb1 \strokec4 \
\cb3     \cf5 \cb3 \strokec5 elif\cf4 \cb3 \strokec4  actual\cf6 \cb3 \strokec6 [\cf4 \cb3 \strokec4 j\cf6 \cb3 \strokec6 ]\cf4 \cb3 \strokec4 >=\cf14 \cb3 \strokec14 8.5\cf6 \cb3 \strokec6 :\cf4 \cb1 \strokec4 \
\cb3       actual\cf6 \cb3 \strokec6 [\cf4 \cb3 \strokec4 j\cf6 \cb3 \strokec6 ]\cf4 \cb3 \strokec4 =\cf14 \cb3 \strokec14 9\cf4 \cb1 \strokec4 \
\
\cb3 unique\cf6 \cb3 \strokec6 ,\cf4 \cb3 \strokec4  counts = np.unique\cf6 \cb3 \strokec6 (\cf4 \cb3 \strokec4 actual\cf6 \cb3 \strokec6 ,\cf4 \cb3 \strokec4  return_counts=\cf12 \cb3 \strokec12 True\cf6 \cb3 \strokec6 )\cf4 \cb1 \strokec4 \
\pard\pardeftab720\partightenfactor0
\cf13 \cb3 \strokec13 dict\cf6 \cb3 \strokec6 (\cf15 \cb3 \strokec15 zip\cf6 \cb3 \strokec6 (\cf4 \cb3 \strokec4 unique\cf6 \cb3 \strokec6 ,\cf4 \cb3 \strokec4  counts\cf6 \cb3 \strokec6 ))\cf4 \cb1 \strokec4 \
\
\pard\pardeftab720\partightenfactor0
\cf4 \cb3 temp=pd.DataFrame\cf6 \cb3 \strokec6 (\cf4 \cb3 \strokec4 temp\cf6 \cb3 \strokec6 ,\cf4 \cb3 \strokec4 columns =\cf6 \cb3 \strokec6 [\cf10 \strokec10 'prediction'\cf6 \strokec6 ])\cf4 \cb1 \strokec4 \
\cb3 actual=pd.DataFrame\cf6 \cb3 \strokec6 (\cf4 \cb3 \strokec4 actual\cf6 \cb3 \strokec6 ,\cf4 \cb3 \strokec4 columns =\cf6 \cb3 \strokec6 [\cf10 \strokec10 'actual'\cf6 \strokec6 ])\cf4 \cb1 \strokec4 \
\
\cb3 target_names = \cf6 \cb3 \strokec6 [\cf10 \strokec10 '0'\cf6 \strokec6 ,\cf4 \cb3 \strokec4  \cf10 \cb3 \strokec10 '4'\cf6 \strokec6 ,\cf10 \strokec10 '5'\cf6 \strokec6 ,\cf10 \strokec10 '6'\cf6 \strokec6 ,\cf10 \strokec10 '8'\cf6 \strokec6 ,\cf10 \strokec10 '9'\cf6 \strokec6 ]\cf4 \cb1 \strokec4 \
\pard\pardeftab720\partightenfactor0
\cf15 \cb3 \strokec15 print\cf6 \cb3 \strokec6 (\cf4 \cb3 \strokec4 classification_report\cf6 \cb3 \strokec6 (\cf4 \cb3 \strokec4 actual\cf6 \cb3 \strokec6 ,\cf4 \cb3 \strokec4  temp\cf6 \cb3 \strokec6 ,\cf4 \cb3 \strokec4  target_names=target_names\cf6 \cb3 \strokec6 ))\cf4 \cb1 \strokec4 \
\
\pard\pardeftab720\partightenfactor0
\cf4 \cb3 confusion_matrix\cf6 \cb3 \strokec6 (\cf4 \cb3 \strokec4 actual\cf6 \cb3 \strokec6 ,\cf4 \cb3 \strokec4  temp\cf6 \cb3 \strokec6 )\cf4 \cb1 \strokec4 \
\
#Get the MSE values\
\cb3 mse_1=mean_squared_error\cf6 \cb3 \strokec6 (\cf4 \cb3 \strokec4 labels_1\cf6 \cb3 \strokec6 ,\cf4 \cb3 \strokec4  predictions_1\cf6 \cb3 \strokec6 )\cf4 \cb1 \strokec4 \
\cb3 rmse_1=np.sqrt\cf6 \cb3 \strokec6 (\cf4 \cb3 \strokec4 mse_1\cf6 \cb3 \strokec6 )\cf4 \cb1 \strokec4 \
\pard\pardeftab720\partightenfactor0
\cf15 \cb3 \strokec15 print\cf6 \cb3 \strokec6 (\cf12 \cb3 \strokec12 f\cf10 \cb3 \strokec10 "\cf6 \strokec6 \{\cf4 \cb3 \strokec4 column_names\cf6 \cb3 \strokec6 [\cf14 \cb3 \strokec14 0\cf6 \cb3 \strokec6 ]\}\cf10 \strokec10  (RMSE): \cf6 \strokec6 \{\cf4 \cb3 \strokec4 rmse_1\cf14 \cb3 \strokec14 :.4f\cf6 \cb3 \strokec6 \}\cf10 \strokec10 "\cf6 \strokec6 )\cf4 \cb1 \strokec4 \
\pard\pardeftab720\partightenfactor0
\cf4 \cb3 mse_2=mean_squared_error\cf6 \cb3 \strokec6 (\cf4 \cb3 \strokec4 labels_2\cf6 \cb3 \strokec6 ,\cf4 \cb3 \strokec4  predictions_2\cf6 \cb3 \strokec6 )\cf4 \cb1 \strokec4 \
\cb3 rmse_2=np.sqrt\cf6 \cb3 \strokec6 (\cf4 \cb3 \strokec4 mse_2\cf6 \cb3 \strokec6 )\cf4 \cb1 \strokec4 \
\pard\pardeftab720\partightenfactor0
\cf15 \cb3 \strokec15 print\cf6 \cb3 \strokec6 (\cf12 \cb3 \strokec12 f\cf10 \cb3 \strokec10 "\cf6 \strokec6 \{\cf4 \cb3 \strokec4 column_names\cf6 \cb3 \strokec6 [\cf14 \cb3 \strokec14 1\cf6 \cb3 \strokec6 ]\}\cf10 \strokec10  (RMSE): \cf6 \strokec6 \{\cf4 \cb3 \strokec4 rmse_2\cf14 \cb3 \strokec14 :.4f\cf6 \cb3 \strokec6 \}\cf10 \strokec10 "\cf6 \strokec6 )\
\
\
\pard\pardeftab720\partightenfactor0
\cf4 \cb3 \strokec4 merged_df.to_csv\cf6 \cb3 \strokec6 (\cf12 \cb3 \strokec12 f\cf10 \cb3 \strokec10 "\cf6 \strokec6 \{\cf4 \cb3 \strokec4 column_names\cf6 \cb3 \strokec6 [\cf14 \cb3 \strokec14 0\cf6 \cb3 \strokec6 ]\}\cf10 \strokec10 _transformer.csv"\cf6 \strokec6 )\cf4 \cb1 \strokec4 \
\
\
\
\
\
\
\
\
\
\
\
\
\
}