{\rtf1\ansi\ansicpg1252\cocoartf2636
\cocoatextscaling0\cocoaplatform0{\fonttbl\f0\fmodern\fcharset0 Courier;}
{\colortbl;\red255\green255\blue255;\red113\green184\blue255;\red23\green23\blue23;\red202\green202\blue202;
\red183\green111\blue179;\red212\green212\blue212;\red183\green111\blue179;\red23\green23\blue23;\red202\green202\blue202;
\red194\green126\blue101;\red89\green156\blue62;\red167\green197\blue152;\red194\green126\blue101;\red70\green137\blue204;
\red67\green192\blue160;\red212\green214\blue154;\red140\green211\blue254;}
{\*\expandedcolortbl;;\cssrgb\c50980\c77647\c100000;\cssrgb\c11765\c11765\c11765;\cssrgb\c83137\c83137\c83137;
\cssrgb\c77255\c52549\c75294;\cssrgb\c86275\c86275\c86275;\cssrgb\c77255\c52549\c75294;\cssrgb\c11765\c11765\c11765;\cssrgb\c83137\c83137\c83137;
\cssrgb\c80784\c56863\c47059;\cssrgb\c41569\c66275\c30980;\cssrgb\c70980\c80784\c65882;\cssrgb\c80784\c56863\c47059;\cssrgb\c33725\c61176\c83922;
\cssrgb\c30588\c78824\c69020;\cssrgb\c86275\c86275\c66667;\cssrgb\c61176\c86275\c99608;}
\margl1440\margr1440\vieww11520\viewh8400\viewkind0
\deftab720
\pard\pardeftab720\partightenfactor0

\f0\fs28 \cf2 \cb3 \expnd0\expndtw0\kerning0
\outl0\strokewidth0 \strokec2 !\cf4 \strokec4 pip install --quiet pytorch-lightning\cb1 \
\cf2 \cb3 \strokec2 !\cf4 \strokec4 pip install --quiet tqdm\cb1 \
\
\
\pard\pardeftab720\partightenfactor0
\cf5 \cb3 \strokec5 import\cf4 \cb3 \strokec4  seaborn \cf5 \cb3 \strokec5 as\cf4 \cb3 \strokec4  sns\cb1 \
\cf5 \cb3 \strokec5 from\cf4 \cb3 \strokec4  pylab \cf5 \cb3 \strokec5 import\cf4 \cb3 \strokec4  rcParams\cb1 \
\cf5 \cb3 \strokec5 import\cf4 \cb3 \strokec4  matplotlib.pyplot \cf5 \cb3 \strokec5 as\cf4 \cb3 \strokec4  plt\cb1 \
\cf5 \cb3 \strokec5 from\cf4 \cb3 \strokec4  matplotlib \cf5 \cb3 \strokec5 import\cf4 \cb3 \strokec4  rc\cb1 \
\cf5 \cb3 \strokec5 import\cf4 \cb3 \strokec4  math\cb1 \
\cf5 \cb3 \strokec5 import\cf4 \cb3 \strokec4  matplotlib\cb1 \
\cf5 \cb3 \strokec5 import\cf4 \cb3 \strokec4  pandas \cf5 \cb3 \strokec5 as\cf4 \cb3 \strokec4  pd\cb1 \
\cf5 \cb3 \strokec5 import\cf4 \cb3 \strokec4  numpy \cf5 \cb3 \strokec5 as\cf4 \cb3 \strokec4  np\cb1 \
\cf5 \cb3 \strokec5 from\cf4 \cb3 \strokec4  tqdm.notebook \cf5 \cb3 \strokec5 import\cf4 \cb3 \strokec4  tqdm\cb1 \
\cf5 \cb3 \strokec5 import\cf4 \cb3 \strokec4  torch\cb1 \
\cf5 \cb3 \strokec5 import\cf4 \cb3 \strokec4  torch.autograd \cf5 \cb3 \strokec5 as\cf4 \cb3 \strokec4  autograd\cb1 \
\cf5 \cb3 \strokec5 import\cf4 \cb3 \strokec4  torch.nn \cf5 \cb3 \strokec5 as\cf4 \cb3 \strokec4  nn\cb1 \
\cf5 \cb3 \strokec5 import\cf4 \cb3 \strokec4  torch.nn.functional \cf5 \cb3 \strokec5 as\cf4 \cb3 \strokec4  F\cb1 \
\cf5 \cb3 \strokec5 import\cf4 \cb3 \strokec4  torch.optim \cf5 \cb3 \strokec5 as\cf4 \cb3 \strokec4  optim\cb1 \
\cf5 \cb3 \strokec5 import\cf4 \cb3 \strokec4  pytorch_lightning \cf5 \cb3 \strokec5 as\cf4 \cb3 \strokec4  pl\cb1 \
\cf5 \cb3 \strokec5 from\cf4 \cb3 \strokec4  pytorch_lightning.callbacks \cf5 \cb3 \strokec5 import\cf4 \cb3 \strokec4  ModelCheckpoint\cf6 \cb3 \strokec6 ,\cf4 \cb3 \strokec4  EarlyStopping\cb1 \
\cf5 \cb3 \strokec5 from\cf4 \cb3 \strokec4  pytorch_lightning.loggers \cf5 \cb3 \strokec5 import\cf4 \cb3 \strokec4  TensorBoardLogger\cb1 \
\cf5 \cb3 \strokec5 from\cf4 \cb3 \strokec4  torch.utils.data \cf5 \cb3 \strokec5 import\cf4 \cb3 \strokec4  Dataset\cf6 \cb3 \strokec6 ,\cf4 \cb3 \strokec4  DataLoader\cb1 \
\cf5 \cb3 \strokec5 from\cf4 \cb3 \strokec4  sklearn.preprocessing \cf5 \cb3 \strokec5 import\cf4 \cb3 \strokec4  MinMaxScaler\cb1 \
\cf5 \cb3 \strokec5 from\cf4 \cb3 \strokec4  collections \cf5 \cb3 \strokec5 import\cf4 \cb3 \strokec4  defaultdict\
\cf5 \cb3 \strokec5 import\cf4 \cb3 \strokec4  nltk\cb1 \
\cf5 \cb3 \strokec5 from\cf4 \cb3 \strokec4  nltk.tokenize \cf5 \cb3 \strokec5 import\cf4 \cb3 \strokec4  word_tokenize\cb1 \
\pard\pardeftab720\partightenfactor0
\cf7 \cb8 \outl0\strokewidth0 import\cf9  shutil\cb1 \
\cf7 \cb8 from\cf9  sklearn.metrics \cf7 import\cf9  mean_squared_error\cb1 \
\pard\pardeftab720\partightenfactor0
\cf4 \outl0\strokewidth0 \strokec4 \
\pard\pardeftab720\partightenfactor0
\cf4 \cb3 df = pd.read_csv\cf6 \cb3 \strokec6 (\cf10 \strokec10 \'93FF_Dataset_6hour_run.csv\'94\cf6 \strokec6 )\
\
\pard\pardeftab720\partightenfactor0
\cf11 \cb3 \strokec11 # Random Seed Pytorch Lightning\cf4 \cb1 \strokec4 \
\pard\pardeftab720\partightenfactor0
\cf4 \cb3 pl.seed_everything\cf6 \cb3 \strokec6 (\cf12 \cb3 \strokec12 42\cf6 \cb3 \strokec6 )\cf4 \cb1 \strokec4 \
\
\pard\pardeftab720\partightenfactor0
\cf11 \cb3 \strokec11 #for i in all column names except \cf13 \cb8 \outl0\strokewidth0 Description column\cf4 \cb1 \outl0\strokewidth0 \strokec4 \
\pard\pardeftab720\partightenfactor0
\cf4 \cb3 column_names=\cf6 \cb3 \strokec6 [\cf10 \strokec10 'LoadCell_R03'\cf6 \strokec6 ,\cf10 \strokec10 'Description'\cf6 \strokec6 ]\cf4 \cb1 \strokec4 \
\
\cb3 df=df\cf6 \cb3 \strokec6 [\cf4 \cb3 \strokec4 column_names\cf6 \cb3 \strokec6 ]\cf4 \cb1 \strokec4 \
\cb3 nltk.download\cf6 \cb3 \strokec6 (\cf10 \strokec10 'punkt'\cf6 \strokec6 )\cf4 \cb3 \strokec4   \cf11 \cb3 \strokec11 # Download the NLTK tokenizer data (if not downloaded)\cf4 \cb1 \strokec4 \
\cb3 df\cf6 \cb3 \strokec6 [\cf10 \strokec10 'Tokenized_Description'\cf6 \strokec6 ]\cf4 \cb3 \strokec4  = df\cf6 \cb3 \strokec6 [\cf10 \strokec10 'Description'\cf6 \strokec6 ]\cf4 \cb3 \strokec4 .apply\cf6 \cb3 \strokec6 (\cf14 \cb3 \strokec14 lambda\cf4 \cb3 \strokec4  x\cf6 \cb3 \strokec6 :\cf4 \cb3 \strokec4  word_tokenize\cf6 \cb3 \strokec6 (\cf15 \cb3 \strokec15 str\cf6 \cb3 \strokec6 (\cf4 \cb3 \strokec4 x\cf6 \cb3 \strokec6 ))\cf4 \cb3 \strokec4  \cf5 \cb3 \strokec5 if\cf4 \cb3 \strokec4  pd.notnull\cf6 \cb3 \strokec6 (\cf4 \cb3 \strokec4 x\cf6 \cb3 \strokec6 )\cf4 \cb3 \strokec4  \cf5 \cb3 \strokec5 else\cf4 \cb3 \strokec4  \cf6 \cb3 \strokec6 [])\cf4 \cb1 \strokec4 \
\
\cb3 result = df\cf6 \cb3 \strokec6 [[\cf4 \cb3 \strokec4 column_names\cf6 \cb3 \strokec6 [\cf12 \cb3 \strokec12 0\cf6 \cb3 \strokec6 ],\cf4 \cb3 \strokec4  \cf10 \cb3 \strokec10 'Tokenized_Description'\cf6 \strokec6 ]]\cf4 \cb3 \strokec4 .values.tolist\cf6 \cb3 \strokec6 ()\cf4 \cb1 \strokec4 \
\
\cb3 all_tokens = \cf15 \cb3 \strokec15 set\cf6 \cb3 \strokec6 ()\cf4 \cb1 \strokec4 \
\cb3 _ = df\cf6 \cb3 \strokec6 [\cf10 \strokec10 'Tokenized_Description'\cf6 \strokec6 ]\cf4 \cb3 \strokec4 .apply\cf6 \cb3 \strokec6 (\cf14 \cb3 \strokec14 lambda\cf4 \cb3 \strokec4  x\cf6 \cb3 \strokec6 :\cf4 \cb3 \strokec4  all_tokens.update\cf6 \cb3 \strokec6 (\cf4 \cb3 \strokec4 x\cf6 \cb3 \strokec6 )\cf4 \cb3 \strokec4  \cf5 \cb3 \strokec5 if\cf4 \cb3 \strokec4  x \cf5 \cb3 \strokec5 else\cf4 \cb3 \strokec4  all_tokens.add\cf6 \cb3 \strokec6 (\cf14 \cb3 \strokec14 None\cf6 \cb3 \strokec6 ))\cf4 \cb1 \strokec4 \
\
\pard\pardeftab720\partightenfactor0
\cf11 \cb3 \strokec11 # List of all different tokens\cf4 \cb1 \strokec4 \
\pard\pardeftab720\partightenfactor0
\cf4 \cb3 different_tokens = \cf15 \cb3 \strokec15 list\cf6 \cb3 \strokec6 (\cf4 \cb3 \strokec4 all_tokens\cf6 \cb3 \strokec6 )\cf4 \cb1 \strokec4 \
\
\cb3 tokens = \cf6 \cb3 \strokec6 \{\cf10 \strokec10 'None'\cf6 \strokec6 :\cf12 \cb3 \strokec12 0.0\cf6 \cb3 \strokec6 ,\cf10 \strokec10 'Nose'\cf6 \strokec6 :\cf12 \cb3 \strokec12 1.0\cf6 \cb3 \strokec6 ,\cf4 \cb3 \strokec4  \cf10 \cb3 \strokec10 'nose'\cf6 \strokec6 :\cf4 \cb3 \strokec4  \cf12 \cb3 \strokec12 2.0\cf6 \cb3 \strokec6 ,\cf4 \cb3 \strokec4  \cf10 \cb3 \strokec10 'Removed'\cf6 \strokec6 :\cf4 \cb3 \strokec4  \cf12 \cb3 \strokec12 3.0\cf6 \cb3 \strokec6 ,\cf4 \cb3 \strokec4  \cf10 \cb3 \strokec10 'crashed'\cf6 \strokec6 :\cf4 \cb3 \strokec4  \cf12 \cb3 \strokec12 4.0\cf6 \cb3 \strokec6 ,\cf4 \cb3 \strokec4  \cf10 \cb3 \strokec10 'R03'\cf6 \strokec6 :\cf4 \cb3 \strokec4  \cf12 \cb3 \strokec12 5.0\cf6 \cb3 \strokec6 ,\cf4 \cb3 \strokec4  \cf10 \cb3 \strokec10 'Nosecone'\cf6 \strokec6 :\cf4 \cb3 \strokec4  \cf12 \cb3 \strokec12 6.0\cf6 \cb3 \strokec6 ,\cf4 \cb3 \strokec4  \cf10 \cb3 \strokec10 'BothBodies'\cf6 \strokec6 :\cf4 \cb3 \strokec4  \cf12 \cb3 \strokec12 7.0\cf6 \cb3 \strokec6 ,\cf4 \cb3 \strokec4  \cf10 \cb3 \strokec10 'R04'\cf6 \strokec6 :\cf4 \cb3 \strokec4  \cf12 \cb3 \strokec12 8.0\cf6 \cb3 \strokec6 ,\cf4 \cb3 \strokec4  \cf10 \cb3 \strokec10 'Door2_TimedOut'\cf6 \strokec6 :\cf4 \cb3 \strokec4  \cf12 \cb3 \strokec12 9.0\cf6 \cb3 \strokec6 ,\cf4 \cb3 \strokec4  \cf10 \cb3 \strokec10 'TopBody'\cf6 \strokec6 :\cf4 \cb3 \strokec4  \cf12 \cb3 \strokec12 10.0\cf6 \cb3 \strokec6 ,\cf4 \cb3 \strokec4  \cf10 \cb3 \strokec10 'and'\cf6 \strokec6 :\cf4 \cb3 \strokec4  \cf12 \cb3 \strokec12 11.0\cf6 \cb3 \strokec6 ,\cf4 \cb3 \strokec4  \cf10 \cb3 \strokec10 'ESTOPPED'\cf6 \strokec6 :\cf4 \cb3 \strokec4  \cf12 \cb3 \strokec12 12.0\cf6 \cb3 \strokec6 ,\cf4 \cb3 \strokec4  \cf10 \cb3 \strokec10 'Body2'\cf6 \strokec6 :\cf4 \cb3 \strokec4  \cf12 \cb3 \strokec12 13.0\cf6 \cb3 \strokec6 ,\cf4 \cb3 \strokec4  \cf10 \cb3 \strokec10 'tail'\cf6 \strokec6 :\cf4 \cb3 \strokec4  \cf12 \cb3 \strokec12 14.0\cf6 \cb3 \strokec6 \}\cf4 \cb1 \strokec4 \
\
\pard\pardeftab720\partightenfactor0
\cf14 \cb3 \strokec14 def\cf4 \cb3 \strokec4  \cf16 \cb3 \strokec16 replace_with_numeric\cf4 \cb3 \strokec4 (\cf17 \cb3 \strokec17 tokens_dict\cf4 \cb3 \strokec4 , \cf17 \cb3 \strokec17 tokens_list\cf4 \cb3 \strokec4 )\cf6 \cb3 \strokec6 :\cf4 \cb1 \strokec4 \
\pard\pardeftab720\partightenfactor0
\cf4 \cb3     \cf5 \cb3 \strokec5 return\cf4 \cb3 \strokec4  \cf6 \cb3 \strokec6 [\cf4 \cb3 \strokec4 tokens_dict\cf6 \cb3 \strokec6 [\cf4 \cb3 \strokec4 token\cf6 \cb3 \strokec6 ]\cf4 \cb3 \strokec4  \cf5 \cb3 \strokec5 if\cf4 \cb3 \strokec4  token \cf2 \strokec2 in\cf4 \strokec4  tokens_dict \cf5 \cb3 \strokec5 else\cf4 \cb3 \strokec4  \cf14 \cb3 \strokec14 None\cf4 \cb3 \strokec4  \cf5 \cb3 \strokec5 for\cf4 \cb3 \strokec4  token \cf2 \strokec2 in\cf4 \strokec4  tokens_list\cf6 \cb3 \strokec6 ]\cf4 \cb1 \strokec4 \
\
\cb3 df\cf6 \cb3 \strokec6 [\cf10 \strokec10 'Tokenized_Description1'\cf6 \strokec6 ]\cf4 \cb3 \strokec4  = df\cf6 \cb3 \strokec6 [\cf10 \strokec10 'Tokenized_Description'\cf6 \strokec6 ]\cf4 \cb3 \strokec4 .apply\cf6 \cb3 \strokec6 (\cf14 \cb3 \strokec14 lambda\cf4 \cb3 \strokec4  x\cf6 \cb3 \strokec6 :\cf4 \cb3 \strokec4  replace_with_numeric\cf6 \cb3 \strokec6 (\cf4 \cb3 \strokec4 tokens\cf6 \cb3 \strokec6 ,\cf4 \cb3 \strokec4  x\cf6 \cb3 \strokec6 ))\cf4 \cb1 \strokec4 \
\
\pard\pardeftab720\partightenfactor0
\cf14 \cb3 \strokec14 def\cf4 \cb3 \strokec4  \cf16 \cb3 \strokec16 calculate_average\cf4 \cb3 \strokec4 (\cf17 \cb3 \strokec17 row\cf4 \cb3 \strokec4 )\cf6 \cb3 \strokec6 :\cf4 \cb1 \strokec4 \
\pard\pardeftab720\partightenfactor0
\cf4 \cb3     numeric_values = \cf6 \cb3 \strokec6 [\cf4 \cb3 \strokec4 value \cf5 \cb3 \strokec5 for\cf4 \cb3 \strokec4  value \cf2 \strokec2 in\cf4 \strokec4  row \cf5 \cb3 \strokec5 if\cf4 \cb3 \strokec4  value \cf2 \strokec2 is\cf4 \strokec4  \cf2 \strokec2 not\cf4 \strokec4  \cf14 \cb3 \strokec14 None\cf6 \cb3 \strokec6 ]\cf4 \cb1 \strokec4 \
\cb3     \cf5 \cb3 \strokec5 return\cf4 \cb3 \strokec4  \cf16 \cb3 \strokec16 sum\cf6 \cb3 \strokec6 (\cf4 \cb3 \strokec4 numeric_values\cf6 \cb3 \strokec6 )\cf4 \cb3 \strokec4  / \cf16 \cb3 \strokec16 len\cf6 \cb3 \strokec6 (\cf4 \cb3 \strokec4 numeric_values\cf6 \cb3 \strokec6 )\cf4 \cb3 \strokec4  \cf5 \cb3 \strokec5 if\cf4 \cb3 \strokec4  \cf16 \cb3 \strokec16 len\cf6 \cb3 \strokec6 (\cf4 \cb3 \strokec4 numeric_values\cf6 \cb3 \strokec6 )\cf4 \cb3 \strokec4  > \cf12 \cb3 \strokec12 0\cf4 \cb3 \strokec4  \cf5 \cb3 \strokec5 else\cf4 \cb3 \strokec4  \cf12 \cb3 \strokec12 0\cf4 \cb1 \strokec4 \
\
\cb3 df\cf6 \cb3 \strokec6 [\cf10 \strokec10 'Average'\cf6 \strokec6 ]\cf4 \cb3 \strokec4  = df\cf6 \cb3 \strokec6 [\cf10 \strokec10 'Tokenized_Description1'\cf6 \strokec6 ]\cf4 \cb3 \strokec4 .apply\cf6 \cb3 \strokec6 (\cf4 \cb3 \strokec4 calculate_average\cf6 \cb3 \strokec6 )\cf4 \cb1 \strokec4 \
\cb3 features_df=df\cf6 \cb3 \strokec6 [[\cf4 \cb3 \strokec4 column_names\cf6 \cb3 \strokec6 [\cf12 \cb3 \strokec12 0\cf6 \cb3 \strokec6 ],\cf10 \strokec10 'Average'\cf6 \strokec6 ]]\cf4 \cb1 \strokec4 \
\cb3 features_df=features_df.rename\cf6 \cb3 \strokec6 (\cf4 \cb3 \strokec4 columns=\cf6 \cb3 \strokec6 \{\cf10 \strokec10 "Average"\cf6 \strokec6 :\cf4 \cb3 \strokec4  \cf10 \cb3 \strokec10 "Description"\cf6 \strokec6 \})\cf4 \cb1 \strokec4 \
\
#train_test split\
\cb3 train_df = features_df\cf6 \cb3 \strokec6 [:\cf15 \cb3 \strokec15 int\cf6 \cb3 \strokec6 (\cf12 \cb3 \strokec12 0.8\cf4 \cb3 \strokec4 *\cf6 \cb3 \strokec6 (\cf16 \cb3 \strokec16 len\cf6 \cb3 \strokec6 (\cf4 \cb3 \strokec4 features_df\cf6 \cb3 \strokec6 )))]\cf4 \cb1 \strokec4 \
\cb3 test_df = features_df\cf6 \cb3 \strokec6 [\cf15 \cb3 \strokec15 int\cf6 \cb3 \strokec6 (\cf12 \cb3 \strokec12 0.8\cf4 \cb3 \strokec4 *\cf6 \cb3 \strokec6 (\cf16 \cb3 \strokec16 len\cf6 \cb3 \strokec6 (\cf4 \cb3 \strokec4 features_df\cf6 \cb3 \strokec6 ))):]\cf4 \cb1 \strokec4 \
\
\pard\pardeftab720\partightenfactor0
\cf11 \cb3 \strokec11 # Normalising the Data\cf4 \cb1 \strokec4 \
\pard\pardeftab720\partightenfactor0
\cf5 \cb3 \strokec5 from\cf4 \cb3 \strokec4  sklearn.preprocessing \cf5 \cb3 \strokec5 import\cf4 \cb3 \strokec4  StandardScaler\cb1 \
\pard\pardeftab720\partightenfactor0
\cf4 \cb3 scaler = StandardScaler\cf6 \cb3 \strokec6 ()\cf4 \cb1 \strokec4 \
\cb3 scalery = scaler.fit\cf6 \cb3 \strokec6 (\cf4 \cb3 \strokec4 train_df\cf6 \cb3 \strokec6 )\cf4 \cb1 \strokec4 \
\
\cb3 train_df = pd.DataFrame\cf6 \cb3 \strokec6 (\cf4 \cb1 \strokec4 \
\cb3     scalery.transform\cf6 \cb3 \strokec6 (\cf4 \cb3 \strokec4 train_df\cf6 \cb3 \strokec6 ),\cf4 \cb1 \strokec4 \
\cb3     index = train_df.index\cf6 \cb3 \strokec6 ,\cf4 \cb1 \strokec4 \
\cb3     columns = train_df.columns\cb1 \
\pard\pardeftab720\partightenfactor0
\cf6 \cb3 \strokec6 )\cf4 \cb1 \strokec4 \
\pard\pardeftab720\partightenfactor0
\cf4 \cb3 train_df.head\cf6 \cb3 \strokec6 ()\cf4 \cb1 \strokec4 \
\
\cb3 test_df = pd.DataFrame\cf6 \cb3 \strokec6 (\cf4 \cb1 \strokec4 \
\cb3     scalery.transform\cf6 \cb3 \strokec6 (\cf4 \cb3 \strokec4 test_df\cf6 \cb3 \strokec6 ),\cf4 \cb1 \strokec4 \
\cb3     index = test_df.index\cf6 \cb3 \strokec6 ,\cf4 \cb1 \strokec4 \
\cb3     columns = test_df.columns\cb1 \
\pard\pardeftab720\partightenfactor0
\cf6 \cb3 \strokec6 )\cf4 \cb1 \strokec4 \
\pard\pardeftab720\partightenfactor0
\cf4 \cb3 test_df.head\cf6 \cb3 \strokec6 ()\cf4 \cb1 \strokec4 \
\
#Generating dataframes in multiple sequences\
\pard\pardeftab720\partightenfactor0
\cf14 \cb3 \strokec14 def\cf4 \cb3 \strokec4  \cf16 \cb3 \strokec16 create_sequences\cf4 \cb3 \strokec4 (\cf17 \cb3 \strokec17 input_data\cf4 \cb3 \strokec4 : \cf17 \cb3 \strokec17 pd\cf4 \cb3 \strokec4 .\cf17 \cb3 \strokec17 DataFrame\cf4 \cb3 \strokec4 , \cf17 \cb3 \strokec17 target_columns\cf4 \cb3 \strokec4 , \cf17 \cb3 \strokec17 sequence_length\cf4 \cb3 \strokec4 =\cf12 \cb3 \strokec12 3\cf4 \cb3 \strokec4 )\cf6 \cb3 \strokec6 :\cf4 \cb1 \strokec4 \
\pard\pardeftab720\partightenfactor0
\cf4 \cb3     sequences = \cf6 \cb3 \strokec6 []\cf4 \cb1 \strokec4 \
\cb3     data_size = \cf16 \cb3 \strokec16 len\cf6 \cb3 \strokec6 (\cf4 \cb3 \strokec4 input_data\cf6 \cb3 \strokec6 )\cf4 \cb1 \strokec4 \
\
\cb3     \cf5 \cb3 \strokec5 for\cf4 \cb3 \strokec4  i \cf2 \strokec2 in\cf4 \strokec4  tqdm\cf6 \cb3 \strokec6 (\cf16 \cb3 \strokec16 range\cf6 \cb3 \strokec6 (\cf4 \cb3 \strokec4 data_size - sequence_length\cf6 \cb3 \strokec6 )):\cf4 \cb1 \strokec4 \
\cb3         sequence = input_data.iloc\cf6 \cb3 \strokec6 [\cf4 \cb3 \strokec4 i\cf6 \cb3 \strokec6 :\cf4 \cb3 \strokec4 i+sequence_length\cf6 \cb3 \strokec6 ]\cf4 \cb1 \strokec4 \
\
\cb3         label_position = i + sequence_length\cb1 \
\cb3         labels = input_data.iloc\cf6 \cb3 \strokec6 [\cf4 \cb3 \strokec4 label_position\cf6 \cb3 \strokec6 ][\cf4 \cb3 \strokec4 target_columns\cf6 \cb3 \strokec6 ]\cf4 \cb1 \strokec4 \
\
\cb3         sequences.append\cf6 \cb3 \strokec6 ((\cf4 \cb3 \strokec4 sequence\cf6 \cb3 \strokec6 ,\cf4 \cb3 \strokec4  labels\cf6 \cb3 \strokec6 ))\cf4 \cb1 \strokec4 \
\
\cb3     \cf5 \cb3 \strokec5 return\cf4 \cb3 \strokec4  sequences\cb1 \
\
\pard\pardeftab720\partightenfactor0
\cf2 \cb3 \strokec2 #Creating Training and Testing Sequences\cf4 \cb1 \strokec4 \
\pard\pardeftab720\partightenfactor0
\cf4 \cb3 SEQUENCE_LENGTH = \cf12 \cb3 \strokec12 120\cf4 \cb1 \strokec4 \
\cb3 target_columns = column_names\
\pard\pardeftab720\partightenfactor0
\cf4 \cb1 \
\pard\pardeftab720\partightenfactor0
\cf4 \cb3 train_sequences = create_sequences\cf6 \cb3 \strokec6 (\cf4 \cb3 \strokec4 train_df\cf6 \cb3 \strokec6 ,\cf4 \cb3 \strokec4  target_columns\cf6 \cb3 \strokec6 ,\cf4 \cb3 \strokec4  sequence_length=SEQUENCE_LENGTH\cf6 \cb3 \strokec6 )\cf4 \cb1 \strokec4 \
\cb3 test_sequences = create_sequences\cf6 \cb3 \strokec6 (\cf4 \cb3 \strokec4 test_df\cf6 \cb3 \strokec6 ,\cf4 \cb3 \strokec4  target_columns\cf6 \cb3 \strokec6 ,\cf4 \cb3 \strokec4  sequence_length=SEQUENCE_LENGTH\cf6 \cb3 \strokec6 )\
\
\pard\pardeftab720\partightenfactor0
\cf11 \cb3 \strokec11 # To check sequence, label and shape\cf4 \cb1 \strokec4 \
\pard\pardeftab720\partightenfactor0
\cf16 \cb3 \strokec16 print\cf6 \cb3 \strokec6 (\cf10 \strokec10 "Label: "\cf6 \strokec6 ,\cf4 \cb3 \strokec4  train_sequences\cf6 \cb3 \strokec6 [\cf12 \cb3 \strokec12 0\cf6 \cb3 \strokec6 ][\cf12 \cb3 \strokec12 1\cf6 \cb3 \strokec6 ])\cf4 \cb1 \strokec4 \
\cf16 \cb3 \strokec16 print\cf6 \cb3 \strokec6 (\cf10 \strokec10 ""\cf6 \strokec6 )\cf4 \cb1 \strokec4 \
\cf16 \cb3 \strokec16 print\cf6 \cb3 \strokec6 (\cf10 \strokec10 "Sequence: "\cf6 \strokec6 ,\cf4 \cb3 \strokec4 train_sequences\cf6 \cb3 \strokec6 [\cf12 \cb3 \strokec12 0\cf6 \cb3 \strokec6 ][\cf12 \cb3 \strokec12 0\cf6 \cb3 \strokec6 ])\cf4 \cb1 \strokec4 \
\cf16 \cb3 \strokec16 print\cf6 \cb3 \strokec6 (\cf10 \strokec10 "Sequence Shape: "\cf6 \strokec6 ,\cf4 \cb3 \strokec4 train_sequences\cf6 \cb3 \strokec6 [\cf12 \cb3 \strokec12 0\cf6 \cb3 \strokec6 ][\cf12 \cb3 \strokec12 0\cf6 \cb3 \strokec6 ]\cf4 \cb3 \strokec4 .shape\cf6 \cb3 \strokec6 )\
\
\
#Creating PyTorch Datasets\cf4 \cb1 \strokec4 \
\pard\pardeftab720\partightenfactor0
\cf14 \cb3 \strokec14 class\cf4 \cb3 \strokec4  \cf15 \cb3 \strokec15 FFDataset\cf4 \cb3 \strokec4 (\cf15 \cb3 \strokec15 Dataset\cf4 \cb3 \strokec4 )\cf6 \cb3 \strokec6 :\cf4 \cb1 \strokec4 \
\pard\pardeftab720\partightenfactor0
\cf4 \cb3   \cf14 \cb3 \strokec14 def\cf4 \cb3 \strokec4  \cf16 \cb3 \strokec16 __init__\cf4 \cb3 \strokec4 (\cf17 \cb3 \strokec17 self\cf4 \cb3 \strokec4 , \cf17 \cb3 \strokec17 sequences\cf4 \cb3 \strokec4 )\cf6 \cb3 \strokec6 :\cf4 \cb1 \strokec4 \
\cb3     \cf17 \cb3 \strokec17 self\cf4 \cb3 \strokec4 .sequences = sequences\cb1 \
\
\cb3   \cf14 \cb3 \strokec14 def\cf4 \cb3 \strokec4  \cf16 \cb3 \strokec16 __len__\cf4 \cb3 \strokec4 (\cf17 \cb3 \strokec17 self\cf4 \cb3 \strokec4 )\cf6 \cb3 \strokec6 :\cf4 \cb1 \strokec4 \
\cb3     \cf5 \cb3 \strokec5 return\cf4 \cb3 \strokec4  \cf16 \cb3 \strokec16 len\cf6 \cb3 \strokec6 (\cf17 \cb3 \strokec17 self\cf4 \cb3 \strokec4 .sequences\cf6 \cb3 \strokec6 )\cf4 \cb1 \strokec4 \
\
\cb3   \cf14 \cb3 \strokec14 def\cf4 \cb3 \strokec4  \cf16 \cb3 \strokec16 __getitem__\cf4 \cb3 \strokec4 (\cf17 \cb3 \strokec17 self\cf4 \cb3 \strokec4 , \cf17 \cb3 \strokec17 idx\cf4 \cb3 \strokec4 )\cf6 \cb3 \strokec6 :\cf4 \cb1 \strokec4 \
\cb3     sequence\cf6 \cb3 \strokec6 ,\cf4 \cb3 \strokec4  label = \cf17 \cb3 \strokec17 self\cf4 \cb3 \strokec4 .sequences\cf6 \cb3 \strokec6 [\cf4 \cb3 \strokec4 idx\cf6 \cb3 \strokec6 ]\cf4 \cb1 \strokec4 \
\cb3     \cf5 \cb3 \strokec5 return\cf4 \cb3 \strokec4  \cf15 \cb3 \strokec15 dict\cf6 \cb3 \strokec6 (\cf4 \cb1 \strokec4 \
\cb3         sequence = torch.Tensor\cf6 \cb3 \strokec6 (\cf4 \cb3 \strokec4 sequence.to_numpy\cf6 \cb3 \strokec6 ()),\cf4 \cb1 \strokec4 \
\cb3         label = torch.tensor\cf6 \cb3 \strokec6 (\cf4 \cb3 \strokec4 label\cf6 \cb3 \strokec6 )\cf4 \cb3 \strokec4 .\cf15 \cb3 \strokec15 float\cf6 \cb3 \strokec6 ()\cf4 \cb1 \strokec4 \
\cb3     \cf6 \cb3 \strokec6 )\cf4 \cb1 \strokec4 \
\
\pard\pardeftab720\partightenfactor0
\cf14 \cb3 \strokec14 class\cf4 \cb3 \strokec4  \cf15 \cb3 \strokec15 FFDataModule\cf4 \cb3 \strokec4 (\cf15 \cb3 \strokec15 pl\cf4 \cb3 \strokec4 .\cf15 \cb3 \strokec15 LightningDataModule\cf4 \cb3 \strokec4 )\cf6 \cb3 \strokec6 :\cf4 \cb1 \strokec4 \
\pard\pardeftab720\partightenfactor0
\cf4 \cb3   \cf14 \cb3 \strokec14 def\cf4 \cb3 \strokec4  \cf16 \cb3 \strokec16 __init__\cf4 \cb3 \strokec4 (\cb1 \
\cb3       \cf17 \cb3 \strokec17 self\cf4 \cb3 \strokec4 , \cf17 \cb3 \strokec17 train_sequences\cf4 \cb3 \strokec4 , \cf17 \cb3 \strokec17 test_sequences\cf4 \cb3 \strokec4 , \cf17 \cb3 \strokec17 batch_size\cf4 \cb3 \strokec4  = \cf12 \cb3 \strokec12 8\cf4 \cb1 \strokec4 \
\cb3   )\cf6 \cb3 \strokec6 :\cf4 \cb1 \strokec4 \
\cb3     super\cf6 \cb3 \strokec6 ()\cf4 \cb3 \strokec4 .\cf16 \cb3 \strokec16 __init__\cf6 \cb3 \strokec6 ()\cf4 \cb1 \strokec4 \
\cb3     \cf17 \cb3 \strokec17 self\cf4 \cb3 \strokec4 .train_sequences = train_sequences\cb1 \
\cb3     \cf17 \cb3 \strokec17 self\cf4 \cb3 \strokec4 .test_sequences = test_sequences\cb1 \
\cb3     \cf17 \cb3 \strokec17 self\cf4 \cb3 \strokec4 .batch_size = batch_size\cb1 \
\
\cb3   \cf14 \cb3 \strokec14 def\cf4 \cb3 \strokec4  \cf16 \cb3 \strokec16 setup\cf4 \cb3 \strokec4 (\cf17 \cb3 \strokec17 self\cf4 \cb3 \strokec4 , \cf17 \cb3 \strokec17 stage\cf4 \cb3 \strokec4 =\cf14 \cb3 \strokec14 None\cf4 \cb3 \strokec4 )\cf6 \cb3 \strokec6 :\cf4 \cb1 \strokec4 \
\cb3     \cf17 \cb3 \strokec17 self\cf4 \cb3 \strokec4 .train_dataset = FFDataset\cf6 \cb3 \strokec6 (\cf17 \cb3 \strokec17 self\cf4 \cb3 \strokec4 .train_sequences\cf6 \cb3 \strokec6 )\cf4 \cb1 \strokec4 \
\cb3     \cf17 \cb3 \strokec17 self\cf4 \cb3 \strokec4 .test_dataset = FFDataset\cf6 \cb3 \strokec6 (\cf17 \cb3 \strokec17 self\cf4 \cb3 \strokec4 .test_sequences\cf6 \cb3 \strokec6 )\cf4 \cb1 \strokec4 \
\
\cb3   \cf14 \cb3 \strokec14 def\cf4 \cb3 \strokec4  \cf16 \cb3 \strokec16 train_dataloader\cf4 \cb3 \strokec4 (\cf17 \cb3 \strokec17 self\cf4 \cb3 \strokec4 )\cf6 \cb3 \strokec6 :\cf4 \cb1 \strokec4 \
\cb3     \cf5 \cb3 \strokec5 return\cf4 \cb3 \strokec4  DataLoader\cf6 \cb3 \strokec6 (\cf4 \cb1 \strokec4 \
\cb3         \cf17 \cb3 \strokec17 self\cf4 \cb3 \strokec4 .train_dataset\cf6 \cb3 \strokec6 ,\cf4 \cb1 \strokec4 \
\cb3         batch_size = \cf17 \cb3 \strokec17 self\cf4 \cb3 \strokec4 .batch_size\cf6 \cb3 \strokec6 ,\cf4 \cb1 \strokec4 \
\cb3         shuffle = \cf14 \cb3 \strokec14 False\cf6 \cb3 \strokec6 ,\cf4 \cb1 \strokec4 \
\cb3         num_workers = \cf12 \cb3 \strokec12 2\cf4 \cb1 \strokec4 \
\cb3     \cf6 \cb3 \strokec6 )\cf4 \cb1 \strokec4 \
\
\cb3   \cf14 \cb3 \strokec14 def\cf4 \cb3 \strokec4  \cf16 \cb3 \strokec16 val_dataloader\cf4 \cb3 \strokec4 (\cf17 \cb3 \strokec17 self\cf4 \cb3 \strokec4 )\cf6 \cb3 \strokec6 :\cf4 \cb1 \strokec4 \
\cb3     \cf5 \cb3 \strokec5 return\cf4 \cb3 \strokec4  DataLoader\cf6 \cb3 \strokec6 (\cf4 \cb1 \strokec4 \
\cb3         \cf17 \cb3 \strokec17 self\cf4 \cb3 \strokec4 .test_dataset\cf6 \cb3 \strokec6 ,\cf4 \cb1 \strokec4 \
\cb3         batch_size = \cf12 \cb3 \strokec12 1\cf6 \cb3 \strokec6 ,\cf4 \cb1 \strokec4 \
\cb3         shuffle = \cf14 \cb3 \strokec14 False\cf6 \cb3 \strokec6 ,\cf4 \cb1 \strokec4 \
\cb3         num_workers = \cf12 \cb3 \strokec12 1\cf4 \cb1 \strokec4 \
\cb3     \cf6 \cb3 \strokec6 )\cf4 \cb1 \strokec4 \
\cb3   \cf14 \cb3 \strokec14 def\cf4 \cb3 \strokec4  \cf16 \cb3 \strokec16 test_dataloader\cf4 \cb3 \strokec4 (\cf17 \cb3 \strokec17 self\cf4 \cb3 \strokec4 )\cf6 \cb3 \strokec6 :\cf4 \cb1 \strokec4 \
\cb3     \cf5 \cb3 \strokec5 return\cf4 \cb3 \strokec4  DataLoader\cf6 \cb3 \strokec6 (\cf4 \cb1 \strokec4 \
\cb3         \cf17 \cb3 \strokec17 self\cf4 \cb3 \strokec4 .test_dataset\cf6 \cb3 \strokec6 ,\cf4 \cb1 \strokec4 \
\cb3         batch_size = \cf12 \cb3 \strokec12 1\cf6 \cb3 \strokec6 ,\cf4 \cb1 \strokec4 \
\cb3         shuffle = \cf14 \cb3 \strokec14 False\cf6 \cb3 \strokec6 ,\cf4 \cb1 \strokec4 \
\cb3         num_workers = \cf12 \cb3 \strokec12 1\cf4 \cb1 \strokec4 \
\cb3     \cf6 \cb3 \strokec6 )\cf4 \cb1 \strokec4 \
\
#Model parameters\
\cb3 N_EPOCHS = \cf12 \cb3 \strokec12 5\cf4 \cb1 \strokec4 \
\cb3 BATCH_SIZE = \cf12 \cb3 \strokec12 64\cf4 \cb1 \strokec4 \
\
\cb3 data_module = FFDataModule\cf6 \cb3 \strokec6 (\cf4 \cb3 \strokec4 train_sequences\cf6 \cb3 \strokec6 ,\cf4 \cb3 \strokec4  test_sequences\cf6 \cb3 \strokec6 ,\cf4 \cb3 \strokec4  batch_size = BATCH_SIZE\cf6 \cb3 \strokec6 )\cf4 \cb1 \strokec4 \
\cb3 data_module.setup\cf6 \cb3 \strokec6 ()\cf4 \cb1 \strokec4 \
\
\cb3 train_dataset = FFDataset\cf6 \cb3 \strokec6 (\cf4 \cb3 \strokec4 train_sequences\cf6 \cb3 \strokec6 )\cf4 \cb1 \strokec4 \
\
\pard\pardeftab720\partightenfactor0
\cf11 \cb3 \strokec11 # Testing the dataloader\cf4 \cb1 \strokec4 \
\pard\pardeftab720\partightenfactor0
\cf4 \cb3 a = \cf16 \cb3 \strokec16 iter\cf6 \cb3 \strokec6 (\cf4 \cb3 \strokec4 train_dataset\cf6 \cb3 \strokec6 )\cf4 \cb1 \strokec4 \
\cb3 b = \cf16 \cb3 \strokec16 next\cf6 \cb3 \strokec6 (\cf4 \cb3 \strokec4 a\cf6 \cb3 \strokec6 )\cf4 \cb1 \strokec4 \
\pard\pardeftab720\partightenfactor0
\cf16 \cb3 \strokec16 print\cf6 \cb3 \strokec6 (\cf10 \strokec10 "Sequence Shape: "\cf6 \strokec6 ,\cf4 \cb3 \strokec4  b\cf6 \cb3 \strokec6 [\cf10 \strokec10 "sequence"\cf6 \strokec6 ]\cf4 \cb3 \strokec4 .shape\cf6 \cb3 \strokec6 )\cf4 \cb1 \strokec4 \
\cf16 \cb3 \strokec16 print\cf6 \cb3 \strokec6 (\cf10 \strokec10 "Label: \{\} and Label Shape: \{\}"\cf4 \cb3 \strokec4 .\cf16 \cb3 \strokec16 format\cf6 \cb3 \strokec6 (\cf4 \cb3 \strokec4 b\cf6 \cb3 \strokec6 [\cf10 \strokec10 "label"\cf6 \strokec6 ],\cf4 \cb3 \strokec4  b\cf6 \cb3 \strokec6 [\cf10 \strokec10 "label"\cf6 \strokec6 ]\cf4 \cb3 \strokec4 .shape\cf6 \cb3 \strokec6 )\cf4 \cb3 \strokec4  \cf6 \cb3 \strokec6 )\cf4 \cb1 \strokec4 \
\
\
#Model\
\pard\pardeftab720\partightenfactor0
\cf14 \cb3 \strokec14 class\cf4 \cb3 \strokec4  \cf15 \cb3 \strokec15 PredictionModel\cf4 \cb3 \strokec4 (\cf15 \cb3 \strokec15 nn\cf4 \cb3 \strokec4 .\cf15 \cb3 \strokec15 Module\cf4 \cb3 \strokec4 )\cf6 \cb3 \strokec6 :\cf4 \cb1 \strokec4 \
\pard\pardeftab720\partightenfactor0
\cf4 \cb3   \cf14 \cb3 \strokec14 def\cf4 \cb3 \strokec4  \cf16 \cb3 \strokec16 __init__\cf4 \cb3 \strokec4 (\cf17 \cb3 \strokec17 self\cf4 \cb3 \strokec4 , \cf17 \cb3 \strokec17 n_features\cf4 \cb3 \strokec4 , \cf17 \cb3 \strokec17 n_hidden\cf4 \cb3 \strokec4 =\cf12 \cb3 \strokec12 128\cf4 \cb3 \strokec4 , \cf17 \cb3 \strokec17 n_layers\cf4 \cb3 \strokec4 =\cf12 \cb3 \strokec12 2\cf4 \cb3 \strokec4 )\cf6 \cb3 \strokec6 :\cf4 \cb1 \strokec4 \
\cb3     super\cf6 \cb3 \strokec6 ()\cf4 \cb3 \strokec4 .\cf16 \cb3 \strokec16 __init__\cf6 \cb3 \strokec6 ()\cf4 \cb1 \strokec4 \
\
\cb3     \cf17 \cb3 \strokec17 self\cf4 \cb3 \strokec4 .n_hidden = n_hidden\cb1 \
\
\cb3     \cf17 \cb3 \strokec17 self\cf4 \cb3 \strokec4 .lstm = nn.LSTM\cf6 \cb3 \strokec6 (\cf4 \cb1 \strokec4 \
\cb3         input_size = n_features\cf6 \cb3 \strokec6 ,\cf4 \cb1 \strokec4 \
\cb3         hidden_size = n_hidden\cf6 \cb3 \strokec6 ,\cf4 \cb1 \strokec4 \
\cb3         batch_first = \cf14 \cb3 \strokec14 True\cf6 \cb3 \strokec6 ,\cf4 \cb1 \strokec4 \
\cb3         num_layers = n_layers\cf6 \cb3 \strokec6 ,\cf4 \cb3 \strokec4  \cf11 \cb3 \strokec11 # Stack LSTMs\cf4 \cb1 \strokec4 \
\cb3         dropout = \cf12 \cb3 \strokec12 0.2\cf4 \cb1 \strokec4 \
\cb3     \cf6 \cb3 \strokec6 )\cf4 \cb1 \strokec4 \
\
\cb3     \cf17 \cb3 \strokec17 self\cf4 \cb3 \strokec4 .regressor = nn.Linear\cf6 \cb3 \strokec6 (\cf4 \cb3 \strokec4 n_hidden\cf6 \cb3 \strokec6 ,\cf4 \cb3 \strokec4  \cf12 \cb3 \strokec12 2\cf6 \cb3 \strokec6 )\cf4 \cb1 \strokec4 \
\
\cb3   \cf14 \cb3 \strokec14 def\cf4 \cb3 \strokec4  \cf16 \cb3 \strokec16 forward\cf4 \cb3 \strokec4 (\cf17 \cb3 \strokec17 self\cf4 \cb3 \strokec4 , \cf17 \cb3 \strokec17 x\cf4 \cb3 \strokec4 )\cf6 \cb3 \strokec6 :\cf4 \cb1 \strokec4 \
\cb3     \cf17 \cb3 \strokec17 self\cf4 \cb3 \strokec4 .lstm.flatten_parameters\cf6 \cb3 \strokec6 ()\cf4 \cb3 \strokec4   \cf11 \cb3 \strokec11 # For distrubuted training\cf4 \cb1 \strokec4 \
\
\cb3     _\cf6 \cb3 \strokec6 ,\cf4 \cb3 \strokec4  \cf6 \cb3 \strokec6 (\cf4 \cb3 \strokec4 hidden\cf6 \cb3 \strokec6 ,\cf4 \cb3 \strokec4  _\cf6 \cb3 \strokec6 )\cf4 \cb3 \strokec4  = \cf17 \cb3 \strokec17 self\cf4 \cb3 \strokec4 .lstm\cf6 \cb3 \strokec6 (\cf4 \cb3 \strokec4 x\cf6 \cb3 \strokec6 )\cf4 \cb1 \strokec4 \
\cb3     \cf11 \cb3 \strokec11 # We want the output from the last layer to go into the final\cf4 \cb1 \strokec4 \
\cb3     \cf11 \cb3 \strokec11 # regressor linear layer\cf4 \cb1 \strokec4 \
\cb3     out = hidden\cf6 \cb3 \strokec6 [\cf12 \cb3 \strokec12 -1\cf6 \cb3 \strokec6 ]\cf4 \cb1 \strokec4 \
\
\cb3     \cf5 \cb3 \strokec5 return\cf4 \cb3 \strokec4  \cf17 \cb3 \strokec17 self\cf4 \cb3 \strokec4 .regressor\cf6 \cb3 \strokec6 (\cf4 \cb3 \strokec4 out\cf6 \cb3 \strokec6 )\cf4 \cb1 \strokec4 \
\
\
\pard\pardeftab720\partightenfactor0
\cf14 \cb3 \strokec14 class\cf4 \cb3 \strokec4  \cf15 \cb3 \strokec15 Predictor\cf4 \cb3 \strokec4 (\cf15 \cb3 \strokec15 pl\cf4 \cb3 \strokec4 .\cf15 \cb3 \strokec15 LightningModule\cf4 \cb3 \strokec4 )\cf6 \cb3 \strokec6 :\cf4 \cb1 \strokec4 \
\
\pard\pardeftab720\partightenfactor0
\cf4 \cb3   \cf14 \cb3 \strokec14 def\cf4 \cb3 \strokec4  \cf16 \cb3 \strokec16 __init__\cf4 \cb3 \strokec4 (\cf17 \cb3 \strokec17 self\cf4 \cb3 \strokec4 , \cf17 \cb3 \strokec17 n_features\cf4 \cb3 \strokec4 : \cf17 \cb3 \strokec17 int\cf4 \cb3 \strokec4 )\cf6 \cb3 \strokec6 :\cf4 \cb1 \strokec4 \
\cb3     super\cf6 \cb3 \strokec6 ()\cf4 \cb3 \strokec4 .\cf16 \cb3 \strokec16 __init__\cf6 \cb3 \strokec6 ()\cf4 \cb1 \strokec4 \
\cb3     \cf17 \cb3 \strokec17 self\cf4 \cb3 \strokec4 .model = PredictionModel\cf6 \cb3 \strokec6 (\cf4 \cb3 \strokec4 n_features\cf6 \cb3 \strokec6 )\cf4 \cb1 \strokec4 \
\cb3     \cf17 \cb3 \strokec17 self\cf4 \cb3 \strokec4 .criterion = nn.MSELoss\cf6 \cb3 \strokec6 ()\cf4 \cb1 \strokec4 \
\
\cb3   \cf14 \cb3 \strokec14 def\cf4 \cb3 \strokec4  \cf16 \cb3 \strokec16 forward\cf4 \cb3 \strokec4 (\cf17 \cb3 \strokec17 self\cf4 \cb3 \strokec4 , \cf17 \cb3 \strokec17 x\cf4 \cb3 \strokec4 , \cf17 \cb3 \strokec17 labels\cf4 \cb3 \strokec4 =\cf14 \cb3 \strokec14 None\cf4 \cb3 \strokec4 )\cf6 \cb3 \strokec6 :\cf4 \cb1 \strokec4 \
\cb3     output = \cf17 \cb3 \strokec17 self\cf4 \cb3 \strokec4 .model\cf6 \cb3 \strokec6 (\cf4 \cb3 \strokec4 x\cf6 \cb3 \strokec6 )\cf4 \cb1 \strokec4 \
\
\cb3     loss = \cf12 \cb3 \strokec12 0\cf4 \cb1 \strokec4 \
\
\cb3     \cf5 \cb3 \strokec5 if\cf4 \cb3 \strokec4  labels \cf2 \strokec2 is\cf4 \strokec4  \cf2 \strokec2 not\cf4 \strokec4  \cf14 \cb3 \strokec14 None\cf6 \cb3 \strokec6 :\cf4 \cb1 \strokec4 \
\cb3       loss = \cf17 \cb3 \strokec17 self\cf4 \cb3 \strokec4 .criterion\cf6 \cb3 \strokec6 (\cf4 \cb3 \strokec4 output\cf6 \cb3 \strokec6 ,\cf4 \cb3 \strokec4  labels.unsqueeze\cf6 \cb3 \strokec6 (\cf4 \cb3 \strokec4 dim=\cf12 \cb3 \strokec12 1\cf6 \cb3 \strokec6 ))\cf4 \cb1 \strokec4 \
\
\cb3     \cf5 \cb3 \strokec5 return\cf4 \cb3 \strokec4  loss\cf6 \cb3 \strokec6 ,\cf4 \cb3 \strokec4  output\cb1 \
\
\cb3   \cf14 \cb3 \strokec14 def\cf4 \cb3 \strokec4  \cf16 \cb3 \strokec16 training_step\cf4 \cb3 \strokec4 (\cf17 \cb3 \strokec17 self\cf4 \cb3 \strokec4 , \cf17 \cb3 \strokec17 batch\cf4 \cb3 \strokec4 , \cf17 \cb3 \strokec17 batch_idx\cf4 \cb3 \strokec4 )\cf6 \cb3 \strokec6 :\cf4 \cb1 \strokec4 \
\cb3     sequences = batch\cf6 \cb3 \strokec6 [\cf10 \strokec10 "sequence"\cf6 \strokec6 ]\cf4 \cb1 \strokec4 \
\cb3     labels = batch\cf6 \cb3 \strokec6 [\cf10 \strokec10 "label"\cf6 \strokec6 ]\cf4 \cb1 \strokec4 \
\
\cb3     loss\cf6 \cb3 \strokec6 ,\cf4 \cb3 \strokec4  output = \cf17 \cb3 \strokec17 self\cf4 \cb3 \strokec4 .forward\cf6 \cb3 \strokec6 (\cf4 \cb3 \strokec4 sequences\cf6 \cb3 \strokec6 ,\cf4 \cb3 \strokec4  labels\cf6 \cb3 \strokec6 )\cf4 \cb1 \strokec4 \
\
\cb3     \cf17 \cb3 \strokec17 self\cf4 \cb3 \strokec4 .log\cf6 \cb3 \strokec6 (\cf10 \strokec10 "train_loss"\cf6 \strokec6 ,\cf4 \cb3 \strokec4  loss\cf6 \cb3 \strokec6 ,\cf4 \cb3 \strokec4  prog_bar=\cf14 \cb3 \strokec14 True\cf6 \cb3 \strokec6 ,\cf4 \cb3 \strokec4  logger=\cf14 \cb3 \strokec14 True\cf6 \cb3 \strokec6 )\cf4 \cb1 \strokec4 \
\cb3     \cf16 \cb3 \strokec16 print\cf6 \cb3 \strokec6 (\cf10 \strokec10 "train_loss"\cf6 \strokec6 ,\cf4 \cb3 \strokec4  loss\cf6 \cb3 \strokec6 )\cf4 \cb1 \strokec4 \
\cb3     \cf5 \cb3 \strokec5 return\cf4 \cb3 \strokec4  loss\cb1 \
\
\cb3   \cf14 \cb3 \strokec14 def\cf4 \cb3 \strokec4  \cf16 \cb3 \strokec16 validation_step\cf4 \cb3 \strokec4 (\cf17 \cb3 \strokec17 self\cf4 \cb3 \strokec4 , \cf17 \cb3 \strokec17 batch\cf4 \cb3 \strokec4 , \cf17 \cb3 \strokec17 batch_idx\cf4 \cb3 \strokec4 )\cf6 \cb3 \strokec6 :\cf4 \cb1 \strokec4 \
\cb3     sequences = batch\cf6 \cb3 \strokec6 [\cf10 \strokec10 "sequence"\cf6 \strokec6 ]\cf4 \cb1 \strokec4 \
\cb3     labels = batch\cf6 \cb3 \strokec6 [\cf10 \strokec10 "label"\cf6 \strokec6 ]\cf4 \cb1 \strokec4 \
\
\cb3     loss\cf6 \cb3 \strokec6 ,\cf4 \cb3 \strokec4  output = \cf17 \cb3 \strokec17 self\cf4 \cb3 \strokec4 .forward\cf6 \cb3 \strokec6 (\cf4 \cb3 \strokec4 sequences\cf6 \cb3 \strokec6 ,\cf4 \cb3 \strokec4  labels\cf6 \cb3 \strokec6 )\cf4 \cb1 \strokec4 \
\
\cb3     \cf17 \cb3 \strokec17 self\cf4 \cb3 \strokec4 .log\cf6 \cb3 \strokec6 (\cf10 \strokec10 "val_loss"\cf6 \strokec6 ,\cf4 \cb3 \strokec4  loss\cf6 \cb3 \strokec6 ,\cf4 \cb3 \strokec4  prog_bar=\cf14 \cb3 \strokec14 True\cf6 \cb3 \strokec6 ,\cf4 \cb3 \strokec4  logger=\cf14 \cb3 \strokec14 True\cf6 \cb3 \strokec6 )\cf4 \cb1 \strokec4 \
\cb3     \cf16 \cb3 \strokec16 print\cf6 \cb3 \strokec6 (\cf10 \strokec10 "val_loss"\cf6 \strokec6 ,\cf4 \cb3 \strokec4  loss\cf6 \cb3 \strokec6 )\cf4 \cb1 \strokec4 \
\cb3     \cf5 \cb3 \strokec5 return\cf4 \cb3 \strokec4  loss\cb1 \
\
\cb3   \cf14 \cb3 \strokec14 def\cf4 \cb3 \strokec4  \cf16 \cb3 \strokec16 test_step\cf4 \cb3 \strokec4 (\cf17 \cb3 \strokec17 self\cf4 \cb3 \strokec4 , \cf17 \cb3 \strokec17 batch\cf4 \cb3 \strokec4 , \cf17 \cb3 \strokec17 batch_idx\cf4 \cb3 \strokec4 )\cf6 \cb3 \strokec6 :\cf4 \cb1 \strokec4 \
\cb3     sequences = batch\cf6 \cb3 \strokec6 [\cf10 \strokec10 "sequence"\cf6 \strokec6 ]\cf4 \cb1 \strokec4 \
\cb3     labels = batch\cf6 \cb3 \strokec6 [\cf10 \strokec10 "label"\cf6 \strokec6 ]\cf4 \cb1 \strokec4 \
\
\cb3     loss\cf6 \cb3 \strokec6 ,\cf4 \cb3 \strokec4  output = \cf17 \cb3 \strokec17 self\cf4 \cb3 \strokec4 .forward\cf6 \cb3 \strokec6 (\cf4 \cb3 \strokec4 sequences\cf6 \cb3 \strokec6 ,\cf4 \cb3 \strokec4  labels\cf6 \cb3 \strokec6 )\cf4 \cb1 \strokec4 \
\
\cb3     \cf17 \cb3 \strokec17 self\cf4 \cb3 \strokec4 .log\cf6 \cb3 \strokec6 (\cf10 \strokec10 "test_loss"\cf6 \strokec6 ,\cf4 \cb3 \strokec4  loss\cf6 \cb3 \strokec6 ,\cf4 \cb3 \strokec4  prog_bar=\cf14 \cb3 \strokec14 True\cf6 \cb3 \strokec6 ,\cf4 \cb3 \strokec4  logger=\cf14 \cb3 \strokec14 True\cf6 \cb3 \strokec6 )\cf4 \cb1 \strokec4 \
\cb3     \cf16 \cb3 \strokec16 print\cf6 \cb3 \strokec6 (\cf10 \strokec10 "test_loss"\cf6 \strokec6 ,\cf4 \cb3 \strokec4  loss\cf6 \cb3 \strokec6 )\cf4 \cb1 \strokec4 \
\cb3     \cf5 \cb3 \strokec5 return\cf4 \cb3 \strokec4  loss\cb1 \
\
\cb3   \cf14 \cb3 \strokec14 def\cf4 \cb3 \strokec4  \cf16 \cb3 \strokec16 configure_optimizers\cf4 \cb3 \strokec4 (\cf17 \cb3 \strokec17 self\cf4 \cb3 \strokec4 )\cf6 \cb3 \strokec6 :\cf4 \cb1 \strokec4 \
\cb3     \cf5 \cb3 \strokec5 return\cf4 \cb3 \strokec4  optim.AdamW\cf6 \cb3 \strokec6 (\cf17 \cb3 \strokec17 self\cf4 \cb3 \strokec4 .model.parameters\cf6 \cb3 \strokec6 ())\cf4 \cb1 \strokec4 \
\
\
\cb3 n_features = b\cf6 \cb3 \strokec6 [\cf10 \strokec10 "sequence"\cf6 \strokec6 ]\cf4 \cb3 \strokec4 .shape\cf6 \cb3 \strokec6 [\cf12 \cb3 \strokec12 1\cf6 \cb3 \strokec6 ]\cf4 \cb1 \strokec4 \
\cb3 n_features\cb1 \
\
\cb3 model = Predictor\cf6 \cb3 \strokec6 (\cf4 \cb3 \strokec4 n_features = n_features\cf6 \cb3 \strokec6 )\cf4 \cb1 \strokec4 \
\cb3 n_features\cb1 \
\
\
\pard\pardeftab720\partightenfactor0
\cf5 \cb3 \strokec5 for\cf4 \cb3 \strokec4  item \cf2 \strokec2 in\cf4 \strokec4  data_module.train_dataloader\cf6 \cb3 \strokec6 ():\cf4 \cb1 \strokec4 \
\pard\pardeftab720\partightenfactor0
\cf4 \cb3   \cf16 \cb3 \strokec16 print\cf6 \cb3 \strokec6 (\cf4 \cb3 \strokec4 item\cf6 \cb3 \strokec6 [\cf10 \strokec10 "sequence"\cf6 \strokec6 ]\cf4 \cb3 \strokec4 .shape\cf6 \cb3 \strokec6 )\cf4 \cb1 \strokec4 \
\cb3   \cf16 \cb3 \strokec16 print\cf6 \cb3 \strokec6 (\cf4 \cb3 \strokec4 item\cf6 \cb3 \strokec6 [\cf10 \strokec10 "label"\cf6 \strokec6 ]\cf4 \cb3 \strokec4 .shape\cf6 \cb3 \strokec6 )\cf4 \cb1 \strokec4 \
\cb3   \cf5 \cb3 \strokec5 break\cf4 \cb1 \strokec4 \
\
\
\cb3 shutil.rmtree\cf6 \cb3 \strokec6 (\cf10 \strokec10 '/content/lightning_logs'\cf6 \strokec6 )\cf4 \cb1 \strokec4 \
\
\cb3 checkpoint_callback = ModelCheckpoint\cf6 \cb3 \strokec6 (\cf4 \cb1 \strokec4 \
\cb3     dirpath=\cf10 \cb3 \strokec10 "checkpoints"\cf6 \strokec6 ,\cf4 \cb1 \strokec4 \
\cb3     filename=\cf10 \cb3 \strokec10 "best-checkpoint"\cf6 \strokec6 ,\cf4 \cb1 \strokec4 \
\cb3     save_top_k = \cf12 \cb3 \strokec12 1\cf6 \cb3 \strokec6 ,\cf4 \cb1 \strokec4 \
\cb3     verbose = \cf14 \cb3 \strokec14 True\cf6 \cb3 \strokec6 ,\cf4 \cb1 \strokec4 \
\cb3     monitor = \cf10 \cb3 \strokec10 "val_loss"\cf6 \strokec6 ,\cf4 \cb1 \strokec4 \
\cb3     mode = \cf10 \cb3 \strokec10 "min"\cf4 \cb1 \strokec4 \
\pard\pardeftab720\partightenfactor0
\cf6 \cb3 \strokec6 )\cf4 \cb1 \strokec4 \
\
\pard\pardeftab720\partightenfactor0
\cf4 \cb3 logger = TensorBoardLogger\cf6 \cb3 \strokec6 (\cf10 \strokec10 "lightning_logs"\cf6 \strokec6 ,\cf4 \cb3 \strokec4  name = \cf10 \cb3 \strokec10 "btc-price"\cf6 \strokec6 )\cf4 \cb1 \strokec4 \
\
\cb3 early_stopping_callback = EarlyStopping\cf6 \cb3 \strokec6 (\cf4 \cb3 \strokec4 monitor = \cf10 \cb3 \strokec10 "val_loss"\cf6 \strokec6 ,\cf4 \cb3 \strokec4  patience = \cf12 \cb3 \strokec12 2\cf6 \cb3 \strokec6 )\cf4 \cb1 \strokec4 \
\
\
\cb3 trainer = pl.Trainer\cf6 \cb3 \strokec6 (\cf4 \cb1 \strokec4 \
\cb3     logger = logger\cf6 \cb3 \strokec6 ,\cf4 \cb1 \strokec4 \
\cb3     \cf11 \cb3 \strokec11 #checkpoint_callback = checkpoint_callback,\cf4 \cb1 \strokec4 \
\cb3     callbacks = \cf6 \cb3 \strokec6 [\cf4 \cb3 \strokec4 early_stopping_callback\cf6 \cb3 \strokec6 ],\cf4 \cb1 \strokec4 \
\cb3     max_epochs = N_EPOCHS\cf6 \cb3 \strokec6 ,\cf4 \cb1 \strokec4 \
\cb3     \cf11 \cb3 \strokec11 #gpus = 1,\cf4 \cb1 \strokec4 \
\cb3    \cf11 \cb3 \strokec11 # progress_bar_refresh_rate = 30\cf4 \cb1 \strokec4 \
\pard\pardeftab720\partightenfactor0
\cf6 \cb3 \strokec6 )\cf4 \cb1 \strokec4 \
\
\pard\pardeftab720\partightenfactor0
\cf4 \cb3 trainer.fit\cf6 \cb3 \strokec6 (\cf4 \cb3 \strokec4 model\cf6 \cb3 \strokec6 ,\cf4 \cb3 \strokec4  data_module\cf6 \cb3 \strokec6 )\cf4 \cb1 \strokec4 \
\
#Testing the trained model\
\cb3 checkpoint_path = \cf10 \cb3 \strokec10 \'93Add path to .ckpt here\'94\cf4 \cb1 \strokec4 \
\cb3 trained_model = Predictor.load_from_checkpoint\cf6 \cb3 \strokec6 (\cf4 \cb1 \strokec4 \
\cb3     checkpoint_path\cf6 \cb3 \strokec6 ,\cf4 \cb1 \strokec4 \
\cb3     n_features = n_features   \cf11 \cb3 \strokec11 # 2 in this case\cf4 \cb1 \strokec4 \
\pard\pardeftab720\partightenfactor0
\cf6 \cb3 \strokec6 )\
\
\pard\pardeftab720\partightenfactor0
\cf11 \cb3 \strokec11 # Freezing the model for faster predictions\cf4 \cb1 \strokec4 \
\pard\pardeftab720\partightenfactor0
\cf4 \cb3 trained_model.freeze\cf6 \cb3 \strokec6 ()\cf4 \cb1 \strokec4 \
\
#Getting predictions\
\cb3 test_dataset = FFDataset\cf6 \cb3 \strokec6 (\cf4 \cb3 \strokec4 test_sequences\cf6 \cb3 \strokec6 )\cf4 \cb1 \strokec4 \
\cb3 predictions_1 = \cf6 \cb3 \strokec6 []\cf4 \cb3 \strokec4   \cf11 \cb3 \strokec11 # Predictions for the first column\cf4 \cb1 \strokec4 \
\cb3 predictions_2 = \cf6 \cb3 \strokec6 []\cf4 \cb3 \strokec4   \cf11 \cb3 \strokec11 # Predictions for the second column\cf4 \cb1 \strokec4 \
\cb3 labels_1 = \cf6 \cb3 \strokec6 []\cf4 \cb3 \strokec4        \cf11 \cb3 \strokec11 # Labels for the first column\cf4 \cb1 \strokec4 \
\cb3 labels_2 = \cf6 \cb3 \strokec6 []\cf4 \cb3 \strokec4        \cf11 \cb3 \strokec11 # Labels for the second column\cf4 \cb1 \strokec4 \
\
\pard\pardeftab720\partightenfactor0
\cf5 \cb3 \strokec5 for\cf4 \cb3 \strokec4  item \cf2 \strokec2 in\cf4 \strokec4  tqdm\cf6 \cb3 \strokec6 (\cf4 \cb3 \strokec4 test_dataset\cf6 \cb3 \strokec6 ):\cf4 \cb1 \strokec4 \
\pard\pardeftab720\partightenfactor0
\cf4 \cb3     sequence = item\cf6 \cb3 \strokec6 [\cf10 \strokec10 "sequence"\cf6 \strokec6 ]\cf4 \cb1 \strokec4 \
\cb3     label = item\cf6 \cb3 \strokec6 [\cf10 \strokec10 "label"\cf6 \strokec6 ]\cf4 \cb1 \strokec4 \
\cb3     sequence=sequence.to\cf6 \cb3 \strokec6 (\cf4 \cb3 \strokec4 device\cf6 \cb3 \strokec6 )\cf4 \cb1 \strokec4 \
\cb3     _\cf6 \cb3 \strokec6 ,\cf4 \cb3 \strokec4  output = trained_model\cf6 \cb3 \strokec6 (\cf4 \cb3 \strokec4 sequence\cf6 \cb3 \strokec6 )\cf4 \cb1 \strokec4 \
\
\cb3     \cf11 \cb3 \strokec11 # Assuming output is a tensor of shape (2,)\cf4 \cb1 \strokec4 \
\cb3     pred_1\cf6 \cb3 \strokec6 ,\cf4 \cb3 \strokec4  pred_2 = output.tolist\cf6 \cb3 \strokec6 ()\cf4 \cb3 \strokec4   \cf11 \cb3 \strokec11 # Splitting predictions for two columns\cf4 \cb1 \strokec4 \
\cb3     predictions_1.append\cf6 \cb3 \strokec6 (\cf4 \cb3 \strokec4 pred_1\cf6 \cb3 \strokec6 )\cf4 \cb1 \strokec4 \
\cb3     predictions_2.append\cf6 \cb3 \strokec6 (\cf4 \cb3 \strokec4 pred_2\cf6 \cb3 \strokec6 )\cf4 \cb1 \strokec4 \
\
\cb3     label_1\cf6 \cb3 \strokec6 ,\cf4 \cb3 \strokec4  label_2 = label.tolist\cf6 \cb3 \strokec6 ()\cf4 \cb3 \strokec4   \cf11 \cb3 \strokec11 # Splitting labels for two columns\cf4 \cb1 \strokec4 \
\cb3     labels_1.append\cf6 \cb3 \strokec6 (\cf4 \cb3 \strokec4 label_1\cf6 \cb3 \strokec6 )\cf4 \cb1 \strokec4 \
\cb3     labels_2.append\cf6 \cb3 \strokec6 (\cf4 \cb3 \strokec4 label_2\cf6 \cb3 \strokec6 )\cf4 \cb1 \strokec4 \
\
\cb3 test_dataset = FFDataset\cf6 \cb3 \strokec6 (\cf4 \cb3 \strokec4 test_sequences\cf6 \cb3 \strokec6 )\cf4 \cb1 \strokec4 \
\cb3 test_dataset\cb1 \
\cb3 test_sequences = pd.DataFrame\cf6 \cb3 \strokec6 (\cf4 \cb3 \strokec4 test_sequences\cf6 \cb3 \strokec6 ,\cf4 \cb3 \strokec4  columns=\cf6 \cb3 \strokec6 [\cf10 \strokec10 'column_name'\cf6 \strokec6 ,\cf10 \strokec10 'e'\cf6 \strokec6 ])\cf4 \cb1 \strokec4 \
\cb3 test_sequences\cb1 \
\
\
\pard\pardeftab720\partightenfactor0
\cf11 \cb3 \strokec11 # Convert lists to NumPy arrays\cf4 \cb1 \strokec4 \
\pard\pardeftab720\partightenfactor0
\cf4 \cb3 predictions_1 = np.array\cf6 \cb3 \strokec6 (\cf4 \cb3 \strokec4 predictions_1\cf6 \cb3 \strokec6 )\cf4 \cb1 \strokec4 \
\cb3 predictions_2 = np.array\cf6 \cb3 \strokec6 (\cf4 \cb3 \strokec4 predictions_2\cf6 \cb3 \strokec6 )\cf4 \cb1 \strokec4 \
\cb3 labels_1 = np.array\cf6 \cb3 \strokec6 (\cf4 \cb3 \strokec4 labels_1\cf6 \cb3 \strokec6 )\cf4 \cb1 \strokec4 \
\cb3 labels_2 = np.array\cf6 \cb3 \strokec6 (\cf4 \cb3 \strokec4 labels_2\cf6 \cb3 \strokec6 )\
\
\
#Get MSE values\cf4 \cb1 \strokec4 \
\cb3 mse_1=mean_squared_error\cf6 \cb3 \strokec6 (\cf4 \cb3 \strokec4 labels_1\cf6 \cb3 \strokec6 ,\cf4 \cb3 \strokec4  predictions_1\cf6 \cb3 \strokec6 )\cf4 \cb1 \strokec4 \
\cb3 rmse_1=np.sqrt\cf6 \cb3 \strokec6 (\cf4 \cb3 \strokec4 mse_1\cf6 \cb3 \strokec6 )\cf4 \cb1 \strokec4 \
\pard\pardeftab720\partightenfactor0
\cf16 \cb3 \strokec16 print\cf6 \cb3 \strokec6 (\cf14 \cb3 \strokec14 f\cf10 \cb3 \strokec10 "\cf6 \strokec6 \{\cf4 \cb3 \strokec4 column_names\cf6 \cb3 \strokec6 [\cf12 \cb3 \strokec12 0\cf6 \cb3 \strokec6 ]\}\cf10 \strokec10  (RMSE): \cf6 \strokec6 \{\cf4 \cb3 \strokec4 rmse_1\cf12 \cb3 \strokec12 :.4f\cf6 \cb3 \strokec6 \}\cf10 \strokec10 "\cf6 \strokec6 )\cf4 \cb1 \strokec4 \
\cf16 \cb3 \strokec16 print\cf6 \cb3 \strokec6 (\cf14 \cb3 \strokec14 f\cf10 \cb3 \strokec10 "\cf6 \strokec6 \{\cf4 \cb3 \strokec4 column_names\cf6 \cb3 \strokec6 [\cf12 \cb3 \strokec12 0\cf6 \cb3 \strokec6 ]\}\cf10 \strokec10  (MSE): \cf6 \strokec6 \{\cf4 \cb3 \strokec4 mse_1\cf12 \cb3 \strokec12 :.4f\cf6 \cb3 \strokec6 \}\cf10 \strokec10 "\cf6 \strokec6 )\cf4 \cb1 \strokec4 \
\pard\pardeftab720\partightenfactor0
\cf4 \cb3 mse_2=mean_squared_error\cf6 \cb3 \strokec6 (\cf4 \cb3 \strokec4 labels_2\cf6 \cb3 \strokec6 ,\cf4 \cb3 \strokec4  predictions_2\cf6 \cb3 \strokec6 )\cf4 \cb1 \strokec4 \
\cb3 rmse_2=np.sqrt\cf6 \cb3 \strokec6 (\cf4 \cb3 \strokec4 mse_2\cf6 \cb3 \strokec6 )\cf4 \cb1 \strokec4 \
\pard\pardeftab720\partightenfactor0
\cf16 \cb3 \strokec16 print\cf6 \cb3 \strokec6 (\cf14 \cb3 \strokec14 f\cf10 \cb3 \strokec10 "\cf6 \strokec6 \{\cf4 \cb3 \strokec4 column_names\cf6 \cb3 \strokec6 [\cf12 \cb3 \strokec12 1\cf6 \cb3 \strokec6 ]\}\cf10 \strokec10  (RMSE): \cf6 \strokec6 \{\cf4 \cb3 \strokec4 rmse_2\cf12 \cb3 \strokec12 :.4f\cf6 \cb3 \strokec6 \}\cf10 \strokec10 "\cf6 \strokec6 )\cf4 \cb1 \strokec4 \
\cf16 \cb3 \strokec16 print\cf6 \cb3 \strokec6 (\cf14 \cb3 \strokec14 f\cf10 \cb3 \strokec10 "\cf6 \strokec6 \{\cf4 \cb3 \strokec4 column_names\cf6 \cb3 \strokec6 [\cf12 \cb3 \strokec12 1\cf6 \cb3 \strokec6 ]\}\cf10 \strokec10  (MSE): \cf6 \strokec6 \{\cf4 \cb3 \strokec4 mse_2\cf12 \cb3 \strokec12 :.4f\cf6 \cb3 \strokec6 \}\cf10 \strokec10 "\cf6 \strokec6 )\cf4 \cb1 \strokec4 \
\
\pard\pardeftab720\partightenfactor0
\cf4 \cb3 all_label=np.concatenate\cf6 \cb3 \strokec6 ((\cf4 \cb3 \strokec4 labels_1\cf6 \cb3 \strokec6 ,\cf4 \cb3 \strokec4 labels_2\cf6 \cb3 \strokec6 ))\cf4 \cb1 \strokec4 \
\cb3 all_pred=np.concatenate\cf6 \cb3 \strokec6 ((\cf4 \cb3 \strokec4 predictions_1\cf6 \cb3 \strokec6 ,\cf4 \cb3 \strokec4 predictions_2\cf6 \cb3 \strokec6 ))\cf4 \cb1 \strokec4 \
\cb3 mse_all=mean_squared_error\cf6 \cb3 \strokec6 (\cf4 \cb3 \strokec4 all_label\cf6 \cb3 \strokec6 ,\cf4 \cb3 \strokec4  all_pred\cf6 \cb3 \strokec6 )\cf4 \cb1 \strokec4 \
\cb3 rmse_all=np.sqrt\cf6 \cb3 \strokec6 (\cf4 \cb3 \strokec4 mse_all\cf6 \cb3 \strokec6 )\cf4 \cb1 \strokec4 \
\pard\pardeftab720\partightenfactor0
\cf16 \cb3 \strokec16 print\cf6 \cb3 \strokec6 (\cf14 \cb3 \strokec14 f\cf10 \cb3 \strokec10 "All (RMSE): \cf6 \strokec6 \{\cf4 \cb3 \strokec4 rmse_all\cf12 \cb3 \strokec12 :.4f\cf6 \cb3 \strokec6 \}\cf10 \strokec10 "\cf6 \strokec6 )\cf4 \cb1 \strokec4 \
\cf16 \cb3 \strokec16 print\cf6 \cb3 \strokec6 (\cf14 \cb3 \strokec14 f\cf10 \cb3 \strokec10 "All (MSE): \cf6 \strokec6 \{\cf4 \cb3 \strokec4 mse_all\cf12 \cb3 \strokec12 :.4f\cf6 \cb3 \strokec6 \}\cf10 \strokec10 "\cf6 \strokec6 )\cf4 \cb1 \strokec4 \
\
\cf16 \cb3 \strokec16 print\cf6 \cb3 \strokec6 (\cf10 \strokec10 "predicted"\cf6 \strokec6 ,\cf4 \cb3 \strokec4 predictions_1\cf6 \cb3 \strokec6 [\cf12 \cb3 \strokec12 0\cf6 \cb3 \strokec6 ],\cf4 \cb3 \strokec4 predictions_2\cf6 \cb3 \strokec6 [\cf12 \cb3 \strokec12 0\cf6 \cb3 \strokec6 ]\cf4 \cb3 \strokec4  \cf6 \cb3 \strokec6 )\cf4 \cb1 \strokec4 \
\cf16 \cb3 \strokec16 print\cf6 \cb3 \strokec6 (\cf10 \strokec10 "labels"\cf6 \strokec6 ,\cf4 \cb3 \strokec4 labels_1\cf6 \cb3 \strokec6 [\cf12 \cb3 \strokec12 0\cf6 \cb3 \strokec6 ],\cf4 \cb3 \strokec4 labels_2\cf6 \cb3 \strokec6 [\cf12 \cb3 \strokec12 0\cf6 \cb3 \strokec6 ])\cf4 \cb1 \strokec4 \
\
\
#Doing inverse scaling\
\pard\pardeftab720\partightenfactor0
\cf11 \cb3 \strokec11 # Assuming pred1 and pred2 are your prediction arrays\cf4 \cb1 \strokec4 \
\pard\pardeftab720\partightenfactor0
\cf4 \cb3 pred1 = predictions_1.reshape\cf6 \cb3 \strokec6 (\cf12 \cb3 \strokec12 -1\cf6 \cb3 \strokec6 ,\cf4 \cb3 \strokec4  \cf12 \cb3 \strokec12 1\cf6 \cb3 \strokec6 )\cf4 \cb1 \strokec4 \
\cb3 pred2 = labels_1.reshape\cf6 \cb3 \strokec6 (\cf12 \cb3 \strokec12 -1\cf6 \cb3 \strokec6 ,\cf4 \cb3 \strokec4  \cf12 \cb3 \strokec12 1\cf6 \cb3 \strokec6 )\cf4 \cb1 \strokec4 \
\
\pard\pardeftab720\partightenfactor0
\cf11 \cb3 \strokec11 # Descale the predictions\cf4 \cb1 \strokec4 \
\pard\pardeftab720\partightenfactor0
\cf4 \cb3 descaled_pred1 = scalery.inverse_transform\cf6 \cb3 \strokec6 (\cf4 \cb3 \strokec4 np.concatenate\cf6 \cb3 \strokec6 ([\cf4 \cb3 \strokec4 pred1\cf6 \cb3 \strokec6 ,\cf4 \cb3 \strokec4  np.zeros_like\cf6 \cb3 \strokec6 (\cf4 \cb3 \strokec4 pred1\cf6 \cb3 \strokec6 )],\cf4 \cb3 \strokec4  axis=\cf12 \cb3 \strokec12 1\cf6 \cb3 \strokec6 ))[:,\cf4 \cb3 \strokec4  \cf12 \cb3 \strokec12 0\cf6 \cb3 \strokec6 ]\cf4 \cb1 \strokec4 \
\cb3 descaled_pred2 = scalery.inverse_transform\cf6 \cb3 \strokec6 (\cf4 \cb3 \strokec4 np.concatenate\cf6 \cb3 \strokec6 ([\cf4 \cb3 \strokec4 np.zeros_like\cf6 \cb3 \strokec6 (\cf4 \cb3 \strokec4 pred2\cf6 \cb3 \strokec6 ),\cf4 \cb3 \strokec4  pred2\cf6 \cb3 \strokec6 ],\cf4 \cb3 \strokec4  axis=\cf12 \cb3 \strokec12 1\cf6 \cb3 \strokec6 ))[:,\cf4 \cb3 \strokec4  \cf12 \cb3 \strokec12 1\cf6 \cb3 \strokec6 ]\cf4 \cb1 \strokec4 \
\
\cb3 predictions_descaled_1 = descaled_pred1\cb1 \
\cb3 labels_descaled_1 = descaled_pred2\cb1 \
\
\pard\pardeftab720\partightenfactor0
\cf16 \cb3 \strokec16 print\cf6 \cb3 \strokec6 (\cf4 \cb3 \strokec4 predictions_descaled_1\cf6 \cb3 \strokec6 [:\cf12 \cb3 \strokec12 3\cf6 \cb3 \strokec6 ])\cf4 \cb1 \strokec4 \
\cf16 \cb3 \strokec16 print\cf6 \cb3 \strokec6 (\cf4 \cb3 \strokec4 labels_descaled_1\cf6 \cb3 \strokec6 [:\cf12 \cb3 \strokec12 3\cf6 \cb3 \strokec6 ])\cf4 \cb1 \strokec4 \
\
\
\pard\pardeftab720\partightenfactor0
\cf11 \cb3 \strokec11 # Assuming pred1 and pred2 are your prediction arrays\cf4 \cb1 \strokec4 \
\pard\pardeftab720\partightenfactor0
\cf4 \cb3 pred_desc1 = predictions_2.reshape\cf6 \cb3 \strokec6 (\cf12 \cb3 \strokec12 -1\cf6 \cb3 \strokec6 ,\cf4 \cb3 \strokec4  \cf12 \cb3 \strokec12 1\cf6 \cb3 \strokec6 )\cf4 \cb1 \strokec4 \
\cb3 pred_desc2 = labels_2.reshape\cf6 \cb3 \strokec6 (\cf12 \cb3 \strokec12 -1\cf6 \cb3 \strokec6 ,\cf4 \cb3 \strokec4  \cf12 \cb3 \strokec12 1\cf6 \cb3 \strokec6 )\cf4 \cb1 \strokec4 \
\
\pard\pardeftab720\partightenfactor0
\cf11 \cb3 \strokec11 # Descale the predictions\cf4 \cb1 \strokec4 \
\pard\pardeftab720\partightenfactor0
\cf4 \cb3 descaled_pred1 = scalery.inverse_transform\cf6 \cb3 \strokec6 (\cf4 \cb3 \strokec4 np.concatenate\cf6 \cb3 \strokec6 ([\cf4 \cb3 \strokec4 pred1\cf6 \cb3 \strokec6 ,\cf4 \cb3 \strokec4  np.zeros_like\cf6 \cb3 \strokec6 (\cf4 \cb3 \strokec4 pred_desc1\cf6 \cb3 \strokec6 )],\cf4 \cb3 \strokec4  axis=\cf12 \cb3 \strokec12 1\cf6 \cb3 \strokec6 ))[:,\cf4 \cb3 \strokec4  \cf12 \cb3 \strokec12 0\cf6 \cb3 \strokec6 ]\cf4 \cb1 \strokec4 \
\cb3 descaled_pred2 = scalery.inverse_transform\cf6 \cb3 \strokec6 (\cf4 \cb3 \strokec4 np.concatenate\cf6 \cb3 \strokec6 ([\cf4 \cb3 \strokec4 np.zeros_like\cf6 \cb3 \strokec6 (\cf4 \cb3 \strokec4 pred_desc2\cf6 \cb3 \strokec6 ),\cf4 \cb3 \strokec4  pred_desc2\cf6 \cb3 \strokec6 ],\cf4 \cb3 \strokec4  axis=\cf12 \cb3 \strokec12 1\cf6 \cb3 \strokec6 ))[:,\cf4 \cb3 \strokec4  \cf12 \cb3 \strokec12 1\cf6 \cb3 \strokec6 ]\cf4 \cb1 \strokec4 \
\
\cb3 predictions_descaled_2 = descaled_pred1\cb1 \
\cb3 labels_descaled_2 = descaled_pred2\cb1 \
\
\pard\pardeftab720\partightenfactor0
\cf16 \cb3 \strokec16 print\cf6 \cb3 \strokec6 (\cf4 \cb3 \strokec4 predictions_descaled_2\cf6 \cb3 \strokec6 [:\cf12 \cb3 \strokec12 3\cf6 \cb3 \strokec6 ])\cf4 \cb1 \strokec4 \
\cf16 \cb3 \strokec16 print\cf6 \cb3 \strokec6 (\cf4 \cb3 \strokec4 labels_descaled_2\cf6 \cb3 \strokec6 [:\cf12 \cb3 \strokec12 3\cf6 \cb3 \strokec6 ])\cf4 \cb1 \strokec4 \
\
\pard\pardeftab720\partightenfactor0
\cf4 \cb3 vdf_pred_scaled=predictions_1.\cf16 \cb3 \strokec16 round\cf6 \cb3 \strokec6 (\cf12 \cb3 \strokec12 0\cf6 \cb3 \strokec6 )\cf4 \cb1 \strokec4 \
\cb3 vdf_label_scaled=labels_1.\cf16 \cb3 \strokec16 round\cf6 \cb3 \strokec6 (\cf12 \cb3 \strokec12 0\cf6 \cb3 \strokec6 )\cf4 \cb1 \strokec4 \
\cb3 desc_pred_scaled=predictions_2.\cf16 \cb3 \strokec16 round\cf6 \cb3 \strokec6 (\cf12 \cb3 \strokec12 0\cf6 \cb3 \strokec6 )\cf4 \cb1 \strokec4 \
\cb3 desc_label_scaled=labels_2.\cf16 \cb3 \strokec16 round\cf6 \cb3 \strokec6 (\cf12 \cb3 \strokec12 0\cf6 \cb3 \strokec6 )\cf4 \cb1 \strokec4 \
\
\cb3 vdf_pred_descaled=predictions_descaled_1.\cf16 \cb3 \strokec16 round\cf6 \cb3 \strokec6 (\cf12 \cb3 \strokec12 0\cf6 \cb3 \strokec6 )\cf4 \cb1 \strokec4 \
\cb3 vdf_label_descaled=labels_descaled_1.\cf16 \cb3 \strokec16 round\cf6 \cb3 \strokec6 (\cf12 \cb3 \strokec12 0\cf6 \cb3 \strokec6 )\cf4 \cb1 \strokec4 \
\cb3 desc_pred_descaled=predictions_descaled_2.\cf16 \cb3 \strokec16 round\cf6 \cb3 \strokec6 (\cf12 \cb3 \strokec12 0\cf6 \cb3 \strokec6 )\cf4 \cb1 \strokec4 \
\cb3 desc_label_descaled=labels_descaled_2.\cf16 \cb3 \strokec16 round\cf6 \cb3 \strokec6 (\cf12 \cb3 \strokec12 0\cf6 \cb3 \strokec6 )\cf4 \cb1 \strokec4 \
\
\pard\pardeftab720\partightenfactor0
\cf16 \cb3 \strokec16 print\cf6 \cb3 \strokec6 (\cf10 \strokec10 'vdf_pred_scaled '\cf6 \strokec6 ,\cf4 \cb3 \strokec4 vdf_pred_scaled\cf6 \cb3 \strokec6 )\cf4 \cb1 \strokec4 \
\cf16 \cb3 \strokec16 print\cf6 \cb3 \strokec6 (\cf10 \strokec10 'vdf_label_scaled '\cf6 \strokec6 ,\cf4 \cb3 \strokec4 vdf_label_scaled\cf6 \cb3 \strokec6 )\cf4 \cb1 \strokec4 \
\cf16 \cb3 \strokec16 print\cf6 \cb3 \strokec6 (\cf10 \strokec10 'desc_pred_scaled '\cf6 \strokec6 ,\cf4 \cb3 \strokec4 desc_pred_scaled\cf6 \cb3 \strokec6 )\cf4 \cb1 \strokec4 \
\cf16 \cb3 \strokec16 print\cf6 \cb3 \strokec6 (\cf10 \strokec10 'desc_label_scaled '\cf6 \strokec6 ,\cf4 \cb3 \strokec4 desc_label_scaled\cf6 \cb3 \strokec6 )\cf4 \cb1 \strokec4 \
\
\
\cf16 \cb3 \strokec16 print\cf6 \cb3 \strokec6 (\cf10 \strokec10 'vdf_pred_descaled '\cf6 \strokec6 ,\cf4 \cb3 \strokec4 vdf_pred_descaled\cf6 \cb3 \strokec6 )\cf4 \cb1 \strokec4 \
\cf16 \cb3 \strokec16 print\cf6 \cb3 \strokec6 (\cf10 \strokec10 'vdf_label_descaled '\cf6 \strokec6 ,\cf4 \cb3 \strokec4 vdf_label_descaled\cf6 \cb3 \strokec6 )\cf4 \cb1 \strokec4 \
\cf16 \cb3 \strokec16 print\cf6 \cb3 \strokec6 (\cf10 \strokec10 'desc_pred_descaled '\cf6 \strokec6 ,\cf4 \cb3 \strokec4 desc_pred_descaled\cf6 \cb3 \strokec6 )\cf4 \cb1 \strokec4 \
\cf16 \cb3 \strokec16 print\cf6 \cb3 \strokec6 (\cf10 \strokec10 'desc_label_descaled '\cf6 \strokec6 ,\cf4 \cb3 \strokec4 desc_label_descaled\cf6 \cb3 \strokec6 )\cf4 \cb1 \strokec4 \
\
#check unique value counts of predictions\
\pard\pardeftab720\partightenfactor0
\cf4 \cb3 unique\cf6 \cb3 \strokec6 ,\cf4 \cb3 \strokec4  counts = np.unique\cf6 \cb3 \strokec6 (\cf4 \cb3 \strokec4 predictions_2.\cf16 \cb3 \strokec16 round\cf6 \cb3 \strokec6 (\cf12 \cb3 \strokec12 2\cf6 \cb3 \strokec6 ),\cf4 \cb3 \strokec4  return_counts=\cf14 \cb3 \strokec14 True\cf6 \cb3 \strokec6 )\cf4 \cb1 \strokec4 \
\pard\pardeftab720\partightenfactor0
\cf15 \cb3 \strokec15 dict\cf6 \cb3 \strokec6 (\cf16 \cb3 \strokec16 zip\cf6 \cb3 \strokec6 (\cf4 \cb3 \strokec4 unique\cf6 \cb3 \strokec6 ,\cf4 \cb3 \strokec4  counts\cf6 \cb3 \strokec6 ))\
\
\pard\pardeftab720\partightenfactor0
\cf9 \cb1 \outl0\strokewidth0 #check unique value counts of actual values\cf6 \cb3 \outl0\strokewidth0 \strokec6 \
\pard\pardeftab720\partightenfactor0
\cf4 \cb3 \strokec4 unique\cf6 \cb3 \strokec6 ,\cf4 \cb3 \strokec4  counts = np.unique\cf6 \cb3 \strokec6 (\cf4 \cb3 \strokec4 desc_label_descaled\cf6 \cb3 \strokec6 ,\cf4 \cb3 \strokec4  return_counts=\cf14 \cb3 \strokec14 True\cf6 \cb3 \strokec6 )\cf4 \cb1 \strokec4 \
\pard\pardeftab720\partightenfactor0
\cf15 \cb3 \strokec15 dict\cf6 \cb3 \strokec6 (\cf16 \cb3 \strokec16 zip\cf6 \cb3 \strokec6 (\cf4 \cb3 \strokec4 unique\cf6 \cb3 \strokec6 ,\cf4 \cb3 \strokec4  counts\cf6 \cb3 \strokec6 ))\
\
\pard\pardeftab720\partightenfactor0
\cf4 \cb3 \strokec4 desc_pred_scaled = np.where\cf6 \cb3 \strokec6 (\cf4 \cb3 \strokec4 np.isclose\cf6 \cb3 \strokec6 (\cf4 \cb3 \strokec4 desc_pred_scaled\cf6 \cb3 \strokec6 ,\cf4 \cb3 \strokec4  \cf12 \cb3 \strokec12 -0.0\cf6 \cb3 \strokec6 ),\cf4 \cb3 \strokec4  \cf12 \cb3 \strokec12 0\cf6 \cb3 \strokec6 ,\cf4 \cb3 \strokec4  desc_pred_scaled\cf6 \cb3 \strokec6 )\cf4 \cb1 \strokec4 \
\cb3 desc_pred_scaled = np.where\cf6 \cb3 \strokec6 (\cf4 \cb3 \strokec4 desc_pred_scaled == \cf12 \cb3 \strokec12 2\cf6 \cb3 \strokec6 ,\cf4 \cb3 \strokec4  \cf12 \cb3 \strokec12 9\cf6 \cb3 \strokec6 ,\cf4 \cb3 \strokec4  desc_pred_scaled\cf6 \cb3 \strokec6 )\cf4 \cb1 \strokec4 \
\cb3 desc_pred_scaled = np.where\cf6 \cb3 \strokec6 (\cf4 \cb3 \strokec4 desc_pred_scaled == \cf12 \cb3 \strokec12 5\cf6 \cb3 \strokec6 ,\cf4 \cb3 \strokec4  \cf12 \cb3 \strokec12 8\cf6 \cb3 \strokec6 ,\cf4 \cb3 \strokec4  desc_pred_scaled\cf6 \cb3 \strokec6 )\cf4 \cb1 \strokec4 \
\cb3 desc_pred_scaled = np.where\cf6 \cb3 \strokec6 (\cf4 \cb3 \strokec4 desc_pred_scaled == \cf12 \cb3 \strokec12 3\cf6 \cb3 \strokec6 ,\cf4 \cb3 \strokec4  \cf12 \cb3 \strokec12 6\cf6 \cb3 \strokec6 ,\cf4 \cb3 \strokec4  desc_pred_scaled\cf6 \cb3 \strokec6 )\cf4 \cb1 \strokec4 \
\cb3 desc_pred_scaled = np.where\cf6 \cb3 \strokec6 (\cf4 \cb3 \strokec4 desc_pred_scaled == \cf12 \cb3 \strokec12 1\cf6 \cb3 \strokec6 ,\cf4 \cb3 \strokec4  \cf12 \cb3 \strokec12 5\cf6 \cb3 \strokec6 ,\cf4 \cb3 \strokec4  desc_pred_scaled\cf6 \cb3 \strokec6 )\cf4 \cb1 \strokec4 \
\
\cb3 unique\cf6 \cb3 \strokec6 ,\cf4 \cb3 \strokec4  counts = np.unique\cf6 \cb3 \strokec6 (\cf4 \cb3 \strokec4 desc_pred_scaled\cf6 \cb3 \strokec6 ,\cf4 \cb3 \strokec4  return_counts=\cf14 \cb3 \strokec14 True\cf6 \cb3 \strokec6 )\cf4 \cb1 \strokec4 \
\pard\pardeftab720\partightenfactor0
\cf15 \cb3 \strokec15 dict\cf6 \cb3 \strokec6 (\cf16 \cb3 \strokec16 zip\cf6 \cb3 \strokec6 (\cf4 \cb3 \strokec4 unique\cf6 \cb3 \strokec6 ,\cf4 \cb3 \strokec4  counts\cf6 \cb3 \strokec6 ))\cf4 \cb1 \strokec4 \
\
\pard\pardeftab720\partightenfactor0
\cf5 \cb3 \strokec5 from\cf4 \cb3 \strokec4  sklearn.metrics \cf5 \cb3 \strokec5 import\cf4 \cb3 \strokec4   classification_report\cf6 \cb3 \strokec6 ,\cf4 \cb3 \strokec4 confusion_matrix\cb1 \
\pard\pardeftab720\partightenfactor0
\cf4 \cb3 target_names = \cf6 \cb3 \strokec6 [\cf10 \strokec10 '0.0'\cf6 \strokec6 ,\cf4 \cb3 \strokec4  \cf10 \cb3 \strokec10 '4.0'\cf6 \strokec6 ,\cf10 \strokec10 '5.0'\cf6 \strokec6 ,\cf10 \strokec10 '6.0'\cf6 \strokec6 ,\cf10 \strokec10 '8.0'\cf6 \strokec6 ,\cf10 \strokec10 '9.0'\cf6 \strokec6 ]\cf4 \cb1 \strokec4 \
\pard\pardeftab720\partightenfactor0
\cf16 \cb3 \strokec16 print\cf6 \cb3 \strokec6 (\cf4 \cb3 \strokec4 classification_report\cf6 \cb3 \strokec6 (\cf4 \cb3 \strokec4 desc_label_scaled\cf6 \cb3 \strokec6 ,\cf4 \cb3 \strokec4  predictions_2.\cf16 \cb3 \strokec16 round\cf6 \cb3 \strokec6 (\cf12 \cb3 \strokec12 0\cf6 \cb3 \strokec6 ),\cf4 \cb3 \strokec4  target_names=target_names\cf6 \cb3 \strokec6 ))\cf4 \cb1 \strokec4 \
\
\pard\pardeftab720\partightenfactor0
\cf4 \cb3 confusion_matrix\cf6 \cb3 \strokec6 (\cf4 \cb3 \strokec4 desc_label_descaled\cf6 \cb3 \strokec6 ,\cf4 \cb3 \strokec4  desc_pred_scaled\cf6 \cb3 \strokec6 )\cf4 \cb1 \strokec4 \
\
\cb3 temp_desc=predictions_2.\cf16 \cb3 \strokec16 round\cf6 \cb3 \strokec6 (\cf12 \cb3 \strokec12 2\cf6 \cb3 \strokec6 )\cf4 \cb1 \strokec4 \
\
\cb3 unique\cf6 \cb3 \strokec6 ,\cf4 \cb3 \strokec4  counts = np.unique\cf6 \cb3 \strokec6 (\cf4 \cb3 \strokec4 temp_desc\cf6 \cb3 \strokec6 ,\cf4 \cb3 \strokec4  return_counts=\cf14 \cb3 \strokec14 True\cf6 \cb3 \strokec6 )\cf4 \cb1 \strokec4 \
\
\pard\pardeftab720\partightenfactor0
\cf15 \cb3 \strokec15 dict\cf6 \cb3 \strokec6 (\cf16 \cb3 \strokec16 zip\cf6 \cb3 \strokec6 (\cf4 \cb3 \strokec4 unique\cf6 \cb3 \strokec6 ,\cf4 \cb3 \strokec4  counts\cf6 \cb3 \strokec6 ))\cf4 \cb1 \strokec4 \
\
\pard\pardeftab720\partightenfactor0
\cf4 \cb3 x=temp_desc.size\cb1 \
\cb3 x\cb1 \
\
#creating thresholds for different anomaly classes\
\pard\pardeftab720\partightenfactor0
\cf5 \cb3 \strokec5 for\cf4 \cb3 \strokec4  j \cf2 \strokec2 in\cf4 \strokec4  \cf16 \cb3 \strokec16 range\cf6 \cb3 \strokec6 (\cf12 \cb3 \strokec12 0\cf6 \cb3 \strokec6 ,\cf4 \cb3 \strokec4 x\cf6 \cb3 \strokec6 ):\cf4 \cb1 \strokec4 \
\pard\pardeftab720\partightenfactor0
\cf4 \cb3     \cf5 \cb3 \strokec5 if\cf4 \cb3 \strokec4  temp_desc\cf6 \cb3 \strokec6 [\cf4 \cb3 \strokec4 j\cf6 \cb3 \strokec6 ]\cf4 \cb3 \strokec4 < \cf12 \cb3 \strokec12 -0.18\cf6 \cb3 \strokec6 :\cf4 \cb1 \strokec4 \
\cb3       temp_desc\cf6 \cb3 \strokec6 [\cf4 \cb3 \strokec4 j\cf6 \cb3 \strokec6 ]\cf4 \cb3 \strokec4 =\cf12 \cb3 \strokec12 0\cf4 \cb1 \strokec4 \
\cb3     \cf5 \cb3 \strokec5 elif\cf4 \cb3 \strokec4  temp_desc\cf6 \cb3 \strokec6 [\cf4 \cb3 \strokec4 j\cf6 \cb3 \strokec6 ]\cf4 \cb3 \strokec4 >= \cf12 \cb3 \strokec12 -0.18\cf4 \cb3 \strokec4  \cf2 \strokec2 and\cf4 \strokec4  temp_desc\cf6 \cb3 \strokec6 [\cf4 \cb3 \strokec4 j\cf6 \cb3 \strokec6 ]\cf4 \cb3 \strokec4 <\cf12 \cb3 \strokec12 4.35\cf6 \cb3 \strokec6 :\cf4 \cb1 \strokec4 \
\cb3       temp_desc\cf6 \cb3 \strokec6 [\cf4 \cb3 \strokec4 j\cf6 \cb3 \strokec6 ]\cf4 \cb3 \strokec4 =\cf12 \cb3 \strokec12 4\cf4 \cb1 \strokec4 \
\cb3     \cf5 \cb3 \strokec5 elif\cf4 \cb3 \strokec4  temp_desc\cf6 \cb3 \strokec6 [\cf4 \cb3 \strokec4 j\cf6 \cb3 \strokec6 ]\cf4 \cb3 \strokec4 >=\cf12 \cb3 \strokec12 4.35\cf4 \cb3 \strokec4  \cf2 \strokec2 and\cf4 \strokec4  temp_desc\cf6 \cb3 \strokec6 [\cf4 \cb3 \strokec4 j\cf6 \cb3 \strokec6 ]\cf4 \cb3 \strokec4 < \cf12 \cb3 \strokec12 4.53\cf6 \cb3 \strokec6 :\cf4 \cb1 \strokec4 \
\cb3       temp_desc\cf6 \cb3 \strokec6 [\cf4 \cb3 \strokec4 j\cf6 \cb3 \strokec6 ]\cf4 \cb3 \strokec4 =\cf12 \cb3 \strokec12 5\cf4 \cb1 \strokec4 \
\cb3     \cf5 \cb3 \strokec5 elif\cf4 \cb3 \strokec4  temp_desc\cf6 \cb3 \strokec6 [\cf4 \cb3 \strokec4 j\cf6 \cb3 \strokec6 ]\cf4 \cb3 \strokec4 >=\cf12 \cb3 \strokec12 4.53\cf4 \cb3 \strokec4  \cf2 \strokec2 and\cf4 \strokec4  temp_desc\cf6 \cb3 \strokec6 [\cf4 \cb3 \strokec4 j\cf6 \cb3 \strokec6 ]\cf4 \cb3 \strokec4 <\cf12 \cb3 \strokec12 4.63\cf6 \cb3 \strokec6 :\cf4 \cb1 \strokec4 \
\cb3       temp_desc\cf6 \cb3 \strokec6 [\cf4 \cb3 \strokec4 j\cf6 \cb3 \strokec6 ]\cf4 \cb3 \strokec4 =\cf12 \cb3 \strokec12 6\cf4 \cb1 \strokec4 \
\cb3     \cf5 \cb3 \strokec5 elif\cf4 \cb3 \strokec4  temp_desc\cf6 \cb3 \strokec6 [\cf4 \cb3 \strokec4 j\cf6 \cb3 \strokec6 ]\cf4 \cb3 \strokec4 >=\cf12 \cb3 \strokec12 4.63\cf4 \cb3 \strokec4  \cf2 \strokec2 and\cf4 \strokec4  temp_desc\cf6 \cb3 \strokec6 [\cf4 \cb3 \strokec4 j\cf6 \cb3 \strokec6 ]\cf4 \cb3 \strokec4 <\cf12 \cb3 \strokec12 4.76\cf6 \cb3 \strokec6 :\cf4 \cb1 \strokec4 \
\cb3       temp_desc\cf6 \cb3 \strokec6 [\cf4 \cb3 \strokec4 j\cf6 \cb3 \strokec6 ]\cf4 \cb3 \strokec4 =\cf12 \cb3 \strokec12 8\cf4 \cb1 \strokec4 \
\cb3     \cf5 \cb3 \strokec5 elif\cf4 \cb3 \strokec4  temp_desc\cf6 \cb3 \strokec6 [\cf4 \cb3 \strokec4 j\cf6 \cb3 \strokec6 ]\cf4 \cb3 \strokec4 >= \cf12 \cb3 \strokec12 4.76\cf6 \cb3 \strokec6 :\cf4 \cb1 \strokec4 \
\cb3       temp_desc\cf6 \cb3 \strokec6 [\cf4 \cb3 \strokec4 j\cf6 \cb3 \strokec6 ]\cf4 \cb3 \strokec4 =\cf12 \cb3 \strokec12 9\cf4 \cb1 \strokec4 \
\
#check the new distribution \
\cb3 unique\cf6 \cb3 \strokec6 ,\cf4 \cb3 \strokec4  counts = np.unique\cf6 \cb3 \strokec6 (\cf4 \cb3 \strokec4 temp_desc\cf6 \cb3 \strokec6 ,\cf4 \cb3 \strokec4  return_counts=\cf14 \cb3 \strokec14 True\cf6 \cb3 \strokec6 )\cf4 \cb1 \strokec4 \
\pard\pardeftab720\partightenfactor0
\cf15 \cb3 \strokec15 dict\cf6 \cb3 \strokec6 (\cf16 \cb3 \strokec16 zip\cf6 \cb3 \strokec6 (\cf4 \cb3 \strokec4 unique\cf6 \cb3 \strokec6 ,\cf4 \cb3 \strokec4  counts\cf6 \cb3 \strokec6 ))\cf4 \cb1 \strokec4 \
\
\pard\pardeftab720\partightenfactor0
\cf4 \cb3 unique\cf6 \cb3 \strokec6 ,\cf4 \cb3 \strokec4  counts = np.unique\cf6 \cb3 \strokec6 (\cf4 \cb3 \strokec4 desc_label_descaled\cf6 \cb3 \strokec6 ,\cf4 \cb3 \strokec4  return_counts=\cf14 \cb3 \strokec14 True\cf6 \cb3 \strokec6 )\cf4 \cb1 \strokec4 \
\pard\pardeftab720\partightenfactor0
\cf15 \cb3 \strokec15 dict\cf6 \cb3 \strokec6 (\cf16 \cb3 \strokec16 zip\cf6 \cb3 \strokec6 (\cf4 \cb3 \strokec4 unique\cf6 \cb3 \strokec6 ,\cf4 \cb3 \strokec4  counts\cf6 \cb3 \strokec6 ))\
\
\pard\pardeftab720\partightenfactor0
\cf4 \cb3 \strokec4 target_names = \cf6 \cb3 \strokec6 [\cf10 \strokec10 '0.0'\cf6 \strokec6 ,\cf4 \cb3 \strokec4  \cf10 \cb3 \strokec10 '4.0'\cf6 \strokec6 ,\cf10 \strokec10 '5.0'\cf6 \strokec6 ,\cf10 \strokec10 '6.0'\cf6 \strokec6 ,\cf10 \strokec10 '8.0'\cf6 \strokec6 ,\cf10 \strokec10 '9.0'\cf6 \strokec6 ]\cf4 \cb1 \strokec4 \
\pard\pardeftab720\partightenfactor0
\cf16 \cb3 \strokec16 print\cf6 \cb3 \strokec6 (\cf4 \cb3 \strokec4 classification_report\cf6 \cb3 \strokec6 (\cf4 \cb3 \strokec4 desc_label_descaled\cf6 \cb3 \strokec6 ,\cf4 \cb3 \strokec4  temp_desc\cf6 \cb3 \strokec6 ,\cf4 \cb3 \strokec4  target_names=target_names\cf6 \cb3 \strokec6 ))\cf4 \cb1 \strokec4 \
\
\pard\pardeftab720\partightenfactor0
\cf4 \cb3 confusion_matrix\cf6 \cb3 \strokec6 (\cf4 \cb3 \strokec4 desc_label_descaled\cf6 \cb3 \strokec6 ,\cf4 \cb3 \strokec4  temp_desc\cf6 \cb3 \strokec6 )\cf4 \cb1 \strokec4 \
\
\
\cb3 temp_arr = np.array\cf6 \cb3 \strokec6 (\cf4 \cb3 \strokec4 temp_desc\cf6 \cb3 \strokec6 )\cf4 \cb1 \strokec4 \
\
\cb3 labels_1_temp = np.array\cf6 \cb3 \strokec6 (\cf4 \cb3 \strokec4 desc_label_descaled\cf6 \cb3 \strokec6 )\cf4 \cb1 \strokec4 \
\
\cb3 mse_1=mean_squared_error\cf6 \cb3 \strokec6 (\cf4 \cb3 \strokec4 labels_1_temp\cf6 \cb3 \strokec6 ,\cf4 \cb3 \strokec4  temp_arr\cf6 \cb3 \strokec6 )\cf4 \cb1 \strokec4 \
\cb3 rmse_1=np.sqrt\cf6 \cb3 \strokec6 (\cf4 \cb3 \strokec4 mse_1\cf6 \cb3 \strokec6 )\cf4 \cb1 \strokec4 \
\pard\pardeftab720\partightenfactor0
\cf16 \cb3 \strokec16 print\cf6 \cb3 \strokec6 (\cf14 \cb3 \strokec14 f\cf10 \cb3 \strokec10 "\cf6 \strokec6 \{\cf4 \cb3 \strokec4 column_names\cf6 \cb3 \strokec6 [\cf12 \cb3 \strokec12 1\cf6 \cb3 \strokec6 ]\}\cf10 \strokec10  (RMSE): \cf6 \strokec6 \{\cf4 \cb3 \strokec4 rmse_1\cf12 \cb3 \strokec12 :.4f\cf6 \cb3 \strokec6 \}\cf10 \strokec10 "\cf6 \strokec6 )\cf4 \cb1 \strokec4 \
\cf16 \cb3 \strokec16 print\cf6 \cb3 \strokec6 (\cf14 \cb3 \strokec14 f\cf10 \cb3 \strokec10 "\cf6 \strokec6 \{\cf4 \cb3 \strokec4 column_names\cf6 \cb3 \strokec6 [\cf12 \cb3 \strokec12 1\cf6 \cb3 \strokec6 ]\}\cf10 \strokec10  (MSE): \cf6 \strokec6 \{\cf4 \cb3 \strokec4 mse_1\cf12 \cb3 \strokec12 :.4f\cf6 \cb3 \strokec6 \}\cf10 \strokec10 "\cf6 \strokec6 )\cf4 \cb1 \strokec4 \
\pard\pardeftab720\partightenfactor0
\cf4 \cb3 par_arr = np.array\cf6 \cb3 \strokec6 (\cf4 \cb3 \strokec4 vdf_pred_descaled\cf6 \cb3 \strokec6 )\cf4 \cb1 \strokec4 \
\
\cb3 labels_1_pred = np.array\cf6 \cb3 \strokec6 (\cf4 \cb3 \strokec4 desc_label_descaled\cf6 \cb3 \strokec6 )\
\
\
\cf4 \cb3 \strokec4 all_label=np.concatenate\cf6 \cb3 \strokec6 ((\cf4 \cb3 \strokec4 labels_1\cf6 \cb3 \strokec6 ,\cf4 \cb3 \strokec4 labels_1_temp\cf6 \cb3 \strokec6 ))\cf4 \cb1 \strokec4 \
\cb3 all_pred=np.concatenate\cf6 \cb3 \strokec6 ((\cf4 \cb3 \strokec4 predictions_1\cf6 \cb3 \strokec6 ,\cf4 \cb3 \strokec4 temp_arr\cf6 \cb3 \strokec6 ))\cf4 \cb1 \strokec4 \
\cb3 mse_all=mean_squared_error\cf6 \cb3 \strokec6 (\cf4 \cb3 \strokec4 all_label\cf6 \cb3 \strokec6 ,\cf4 \cb3 \strokec4  all_pred\cf6 \cb3 \strokec6 )\cf4 \cb1 \strokec4 \
\cb3 rmse_all=np.sqrt\cf6 \cb3 \strokec6 (\cf4 \cb3 \strokec4 mse_all\cf6 \cb3 \strokec6 )\cf4 \cb1 \strokec4 \
\pard\pardeftab720\partightenfactor0
\cf16 \cb3 \strokec16 print\cf6 \cb3 \strokec6 (\cf14 \cb3 \strokec14 f\cf10 \cb3 \strokec10 "All (RMSE): \cf6 \strokec6 \{\cf4 \cb3 \strokec4 rmse_all\cf12 \cb3 \strokec12 :.4f\cf6 \cb3 \strokec6 \}\cf10 \strokec10 "\cf6 \strokec6 )\cf4 \cb1 \strokec4 \
\cf16 \cb3 \strokec16 print\cf6 \cb3 \strokec6 (\cf14 \cb3 \strokec14 f\cf10 \cb3 \strokec10 "All (MSE): \cf6 \strokec6 \{\cf4 \cb3 \strokec4 mse_all\cf12 \cb3 \strokec12 :.4f\cf6 \cb3 \strokec6 \}\cf10 \strokec10 "\cf6 \strokec6 )\cf4 \cb1 \strokec4 \
\
\pard\pardeftab720\partightenfactor0
\cf4 \cb3 actual=desc_label_descaled.\cf16 \cb3 \strokec16 round\cf6 \cb3 \strokec6 (\cf12 \cb3 \strokec12 0\cf6 \cb3 \strokec6 )\cf4 \cb1 \strokec4 \
\
\cb3 unique\cf6 \cb3 \strokec6 ,\cf4 \cb3 \strokec4  counts = np.unique\cf6 \cb3 \strokec6 (\cf4 \cb3 \strokec4 actual\cf6 \cb3 \strokec6 ,\cf4 \cb3 \strokec4  return_counts=\cf14 \cb3 \strokec14 True\cf6 \cb3 \strokec6 )\cf4 \cb1 \strokec4 \
\pard\pardeftab720\partightenfactor0
\cf15 \cb3 \strokec15 dict\cf6 \cb3 \strokec6 (\cf16 \cb3 \strokec16 zip\cf6 \cb3 \strokec6 (\cf4 \cb3 \strokec4 unique\cf6 \cb3 \strokec6 ,\cf4 \cb3 \strokec4  counts\cf6 \cb3 \strokec6 ))\
\
\pard\pardeftab720\partightenfactor0
\cf4 \cb3 \strokec4 temp=pd.DataFrame\cf6 \cb3 \strokec6 (\cf4 \cb3 \strokec4 desc_pred_scaled\cf6 \cb3 \strokec6 ,\cf4 \cb3 \strokec4 columns =\cf6 \cb3 \strokec6 [\cf10 \strokec10 'prediction'\cf6 \strokec6 ])\cf4 \cb1 \strokec4 \
\cb3 actual=pd.DataFrame\cf6 \cb3 \strokec6 (\cf4 \cb3 \strokec4 actual\cf6 \cb3 \strokec6 ,\cf4 \cb3 \strokec4 columns =\cf6 \cb3 \strokec6 [\cf10 \strokec10 'actual'\cf6 \strokec6 ])\cf4 \cb1 \strokec4 \
\
\cb3 predictions_scaled_2=pd.DataFrame\cf6 \cb3 \strokec6 (\cf4 \cb3 \strokec4 vdf_pred_scaled\cf6 \cb3 \strokec6 ,\cf4 \cb3 \strokec4 columns =\cf6 \cb3 \strokec6 [\cf14 \cb3 \strokec14 f\cf10 \cb3 \strokec10 '\cf6 \strokec6 \{\cf4 \cb3 \strokec4 column_names\cf6 \cb3 \strokec6 [\cf12 \cb3 \strokec12 0\cf6 \cb3 \strokec6 ]\}\cf10 \strokec10 '\cf6 \strokec6 ])\cf4 \cb1 \strokec4 \
\
\cb3 temp=pd.DataFrame\cf6 \cb3 \strokec6 (\cf4 \cb3 \strokec4 temp_desc\cf6 \cb3 \strokec6 ,\cf4 \cb3 \strokec4 columns =\cf6 \cb3 \strokec6 [\cf10 \strokec10 'prediction'\cf6 \strokec6 ])\cf4 \cb1 \strokec4 \
\
\cb3 predictions_descaled_2=pd.DataFrame\cf6 \cb3 \strokec6 (\cf4 \cb3 \strokec4 predictions_descaled_1.\cf16 \cb3 \strokec16 round\cf6 \cb3 \strokec6 (\cf12 \cb3 \strokec12 6\cf6 \cb3 \strokec6 ),\cf4 \cb3 \strokec4 columns =\cf6 \cb3 \strokec6 [\cf14 \cb3 \strokec14 f\cf10 \cb3 \strokec10 '\cf6 \strokec6 \{\cf4 \cb3 \strokec4 column_names\cf6 \cb3 \strokec6 [\cf12 \cb3 \strokec12 0\cf6 \cb3 \strokec6 ]\}\cf10 \strokec10 '\cf6 \strokec6 ])\cf4 \cb1 \strokec4 \
\
\cb3 merged_df = pd.merge\cf6 \cb3 \strokec6 (\cf4 \cb3 \strokec4 predictions_descaled_2\cf6 \cb3 \strokec6 ,\cf4 \cb3 \strokec4  temp\cf6 \cb3 \strokec6 ,\cf4 \cb3 \strokec4  left_index=\cf14 \cb3 \strokec14 True\cf6 \cb3 \strokec6 ,\cf4 \cb3 \strokec4  right_index=\cf14 \cb3 \strokec14 True\cf6 \cb3 \strokec6 )\cf4 \cb1 \strokec4 \
\
\cb3 merged_df.to_csv\cf6 \cb3 \strokec6 ((\cf14 \cb3 \strokec14 f\cf10 \cb3 \strokec10 "\cf6 \strokec6 \{\cf4 \cb3 \strokec4 column_names\cf6 \cb3 \strokec6 [\cf12 \cb3 \strokec12 0\cf6 \cb3 \strokec6 ]\}\cf10 \strokec10 _lstm.csv"\cf6 \strokec6 ))\
\
\
\cf4 \cb3 \strokec4 vdf = pd.read_csv\cf6 \cb3 \strokec6 (\cf14 \cb3 \strokec14 f\cf10 \cb3 \strokec10 "\cf6 \strokec6 \{\cf4 \cb3 \strokec4 column_names\cf6 \cb3 \strokec6 [\cf12 \cb3 \strokec12 0\cf6 \cb3 \strokec6 ]\}\cf10 \strokec10 _lstm.csv"\cf6 \strokec6 )\cf4 \cb1 \strokec4 \
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