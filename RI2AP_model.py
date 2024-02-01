{\rtf1\ansi\ansicpg1252\cocoartf2636
\cocoatextscaling0\cocoaplatform0{\fonttbl\f0\fmodern\fcharset0 Courier;}
{\colortbl;\red255\green255\blue255;\red113\green184\blue255;\red23\green23\blue23;\red202\green202\blue202;
\red183\green111\blue179;\red212\green212\blue212;\red194\green126\blue101;\red89\green156\blue62;\red70\green137\blue204;
\red67\green192\blue160;\red167\green197\blue152;\red212\green214\blue154;\red140\green211\blue254;}
{\*\expandedcolortbl;;\cssrgb\c50980\c77647\c100000;\cssrgb\c11765\c11765\c11765;\cssrgb\c83137\c83137\c83137;
\cssrgb\c77255\c52549\c75294;\cssrgb\c86275\c86275\c86275;\cssrgb\c80784\c56863\c47059;\cssrgb\c41569\c66275\c30980;\cssrgb\c33725\c61176\c83922;
\cssrgb\c30588\c78824\c69020;\cssrgb\c70980\c80784\c65882;\cssrgb\c86275\c86275\c66667;\cssrgb\c61176\c86275\c99608;}
\margl1440\margr1440\vieww16300\viewh11120\viewkind0
\deftab720
\pard\pardeftab720\partightenfactor0

\f0\fs28 \cf2 \cb3 \expnd0\expndtw0\kerning0
!\cf4 pip install --quiet pytorch-lightning\cb1 \
\cf2 \cb3 !\cf4 pip install --quiet tqdm\cb1 \
\
\pard\pardeftab720\partightenfactor0
\cf5 \cb3 import\cf4  nltk\cb1 \
\cf5 \cb3 from\cf4  nltk.tokenize \cf5 import\cf4  word_tokenize\cb1 \
\cf5 \cb3 import\cf4  pandas \cf5 as\cf4  pd\cb1 \
\
\cf5 \cb3 import\cf4  torch\cb1 \
\cf5 \cb3 import\cf4  torch.nn \cf5 as\cf4  nn\cb1 \
\cf5 \cb3 import\cf4  numpy \cf5 as\cf4  np\cb1 \
\cf5 \cb3 from\cf4  torch.utils.data \cf5 import\cf4  DataLoader\cf6 ,\cf4  TensorDataset\cb1 \
\cf5 \cb3 from\cf4  sklearn.preprocessing \cf5 import\cf4  StandardScaler\cb1 \
\cf5 \cb3 from\cf4  torch.optim.lr_scheduler \cf5 import\cf4  ReduceLROnPlateau\cb1 \
\
\cf5 \cb3 import\cf4  pytorch_lightning \cf5 as\cf4  pl\cb1 \
\cf5 \cb3 from\cf4  pytorch_lightning.callbacks \cf5 import\cf4  ModelCheckpoint\cf6 ,\cf4  EarlyStopping\cb1 \
\cf5 \cb3 from\cf4  pytorch_lightning.loggers \cf5 import\cf4  TensorBoardLogger\cb1 \
\cf5 \cb3 import\cf4  pickle\cb1 \
\
\
\pard\pardeftab720\partightenfactor0
\cf4 \cb3 df = pd.read_csv\cf6 (\cf7 \'93FF_Dataset_6hour_run.csv\'94\cf6 )\cf4 \cb1 \
\
\cb3 df=df.loc\cf6 [\cf4 df\cf6 [\cf7 'Description'\cf6 ]\cf4  == \cf7 "BothBodies and Nose Removed"\cf6 ]\cf4 \cb1 \
\cb3 df\cb1 \
\
\
\pard\pardeftab720\partightenfactor0
\cf5 \cb3 import\cf4  nltk\cb1 \
\cf5 \cb3 from\cf4  nltk.tokenize \cf5 import\cf4  word_tokenize\cb1 \
\
\
\pard\pardeftab720\partightenfactor0
\cf8 \cb3 #change column names accordingly\cf4 \cb1 \
\
\cf8 \cb3 #for i in all column names except desc\cf4 \cb1 \
\pard\pardeftab720\partightenfactor0
\cf4 \cb3 column_names=\cf6 [\cf7 'SJointAngle_R03'\cf6 ,\cf7 'Description'\cf6 ]\cf4 \cb1 \
\
\cb3 df=df\cf6 [\cf4 column_names\cf6 ]\cf4 \cb1 \
\cb3 nltk.download\cf6 (\cf7 'punkt'\cf6 )\cf4   \cf8 # Download the NLTK tokenizer data (if not downloaded)\cf4 \cb1 \
\cb3 df\cf6 [\cf7 'Tokenized_Description'\cf6 ]\cf4  = df\cf6 [\cf7 'Description'\cf6 ]\cf4 .apply\cf6 (\cf9 lambda\cf4  x\cf6 :\cf4  word_tokenize\cf6 (\cf10 str\cf6 (\cf4 x\cf6 ))\cf4  \cf5 if\cf4  pd.notnull\cf6 (\cf4 x\cf6 )\cf4  \cf5 else\cf4  \cf6 [])\cf4 \cb1 \
\
\cb3 result = df\cf6 [[\cf4 column_names\cf6 [\cf11 0\cf6 ],\cf4  \cf7 'Tokenized_Description'\cf6 ]]\cf4 .values.tolist\cf6 ()\cf4 \cb1 \
\
\cb3 all_tokens = \cf10 set\cf6 ()\cf4 \cb1 \
\cb3 _ = df\cf6 [\cf7 'Tokenized_Description'\cf6 ]\cf4 .apply\cf6 (\cf9 lambda\cf4  x\cf6 :\cf4  all_tokens.update\cf6 (\cf4 x\cf6 )\cf4  \cf5 if\cf4  x \cf5 else\cf4  all_tokens.add\cf6 (\cf9 None\cf6 ))\cf4 \cb1 \
\
\pard\pardeftab720\partightenfactor0
\cf8 \cb3 # List of all different tokens\cf4 \cb1 \
\pard\pardeftab720\partightenfactor0
\cf4 \cb3 different_tokens = \cf10 list\cf6 (\cf4 all_tokens\cf6 )\cf4 \cb1 \
\
\cb3 tokens = \cf6 \{\cf7 'None'\cf6 :\cf11 0.0\cf6 ,\cf7 'Nose'\cf6 :\cf11 1.0\cf6 ,\cf4  \cf7 'nose'\cf6 :\cf4  \cf11 2.0\cf6 ,\cf4  \cf7 'Removed'\cf6 :\cf4  \cf11 3.0\cf6 ,\cf4  \cf7 'crashed'\cf6 :\cf4  \cf11 4.0\cf6 ,\cf4  \cf7 'R03'\cf6 :\cf4  \cf11 5.0\cf6 ,\cf4  \cf7 'Nosecone'\cf6 :\cf4  \cf11 6.0\cf6 ,\cf4  \cf7 'BothBodies'\cf6 :\cf4  \cf11 7.0\cf6 ,\cf4  \cf7 'R04'\cf6 :\cf4  \cf11 8.0\cf6 ,\cf4  \cf7 'Door2_TimedOut'\cf6 :\cf4  \cf11 9.0\cf6 ,\cf4  \cf7 'TopBody'\cf6 :\cf4  \cf11 10.0\cf6 ,\cf4  \cf7 'and'\cf6 :\cf4  \cf11 11.0\cf6 ,\cf4  \cf7 'ESTOPPED'\cf6 :\cf4  \cf11 12.0\cf6 ,\cf4  \cf7 'Body2'\cf6 :\cf4  \cf11 13.0\cf6 ,\cf4  \cf7 'tail'\cf6 :\cf4  \cf11 14.0\cf6 \}\cf4 \cb1 \
\
\pard\pardeftab720\partightenfactor0
\cf9 \cb3 def\cf4  \cf12 replace_with_numeric\cf4 (\cf13 tokens_dict\cf4 , \cf13 tokens_list\cf4 )\cf6 :\cf4 \cb1 \
\pard\pardeftab720\partightenfactor0
\cf4 \cb3     \cf5 return\cf4  \cf6 [\cf4 tokens_dict\cf6 [\cf4 token\cf6 ]\cf4  \cf5 if\cf4  token \cf2 in\cf4  tokens_dict \cf5 else\cf4  \cf9 None\cf4  \cf5 for\cf4  token \cf2 in\cf4  tokens_list\cf6 ]\cf4 \cb1 \
\
\cb3 df\cf6 [\cf7 'Tokenized_Description1'\cf6 ]\cf4  = df\cf6 [\cf7 'Tokenized_Description'\cf6 ]\cf4 .apply\cf6 (\cf9 lambda\cf4  x\cf6 :\cf4  replace_with_numeric\cf6 (\cf4 tokens\cf6 ,\cf4  x\cf6 ))\cf4 \cb1 \
\
\pard\pardeftab720\partightenfactor0
\cf9 \cb3 def\cf4  \cf12 replace_empty_with_none\cf4 (\cf13 tokens_list\cf4 )\cf6 :\cf4 \cb1 \
\pard\pardeftab720\partightenfactor0
\cf4 \cb3     \cf5 return\cf4  \cf7 'None'\cf4  \cf5 if\cf4  \cf12 len\cf6 (\cf4 tokens_list\cf6 )\cf4  == \cf11 0\cf4  \cf5 else\cf4  tokens_list\cb1 \
\
\cb3 df\cf6 [\cf7 'Tokenized_Description'\cf6 ]\cf4  = df\cf6 [\cf7 'Tokenized_Description'\cf6 ]\cf4 .apply\cf6 (\cf9 lambda\cf4  x\cf6 :\cf4  replace_empty_with_none\cf6 (\cf4 x\cf6 ))\cf4 \cb1 \
\cb3 \
\pard\pardeftab720\partightenfactor0
\cf9 def\cf4  \cf12 calculate_average\cf4 (\cf13 row\cf4 )\cf6 :\cf4 \cb1 \
\pard\pardeftab720\partightenfactor0
\cf4 \cb3     numeric_values = \cf6 [\cf4 value \cf5 for\cf4  value \cf2 in\cf4  row \cf5 if\cf4  value \cf2 is\cf4  \cf2 not\cf4  \cf9 None\cf6 ]\cf4 \cb1 \
\cb3     \cf5 return\cf4  \cf12 sum\cf6 (\cf4 numeric_values\cf6 )\cf4  / \cf12 len\cf6 (\cf4 numeric_values\cf6 )\cf4  \cf5 if\cf4  \cf12 len\cf6 (\cf4 numeric_values\cf6 )\cf4  > \cf11 0\cf4  \cf5 else\cf4  \cf11 0\cf4 \cb1 \
\
\cb3 df\cf6 [\cf7 'Average'\cf6 ]\cf4  = df\cf6 [\cf7 'Tokenized_Description1'\cf6 ]\cf4 .apply\cf6 (\cf4 calculate_average\cf6 )\cf4 \cb1 \
\
\cb3 split_index = \cf10 int\cf6 (\cf11 0.8\cf4  * \cf12 len\cf6 (\cf4 df\cf6 ))\cf4 \cb1 \
\
\pard\pardeftab720\partightenfactor0
\cf8 \cb3 # Split the data into training and validation sets\cf4 \cb1 \
\pard\pardeftab720\partightenfactor0
\cf4 \cb3 df_train = df.iloc\cf6 [:\cf4 split_index\cf6 ]\cf4 \cb1 \
\cb3 df_test = df.iloc\cf6 [\cf4 split_index\cf6 :]\cf4 \cb1 \
\
\cb3 features_df=df\cf6 [[\cf4 column_names\cf6 [\cf11 0\cf6 ],\cf7 'Average'\cf6 ]]\cf4 \cb1 \
\cb3 features_df=features_df.rename\cf6 (\cf4 columns=\cf6 \{\cf7 "Average"\cf6 :\cf4  \cf7 "Description"\cf6 \})\cf4 \cb1 \
\cb3 \
p_y=features_df.values.tolist\cf6 ()\cf4 \cb1 \
\
\cb3 result = df\cf6 [[\cf4 column_names\cf6 [\cf11 0\cf6 ],\cf7 'Tokenized_Description'\cf6 ]]\cf4 \cb1 \
\cb3 result\cb1 \
\
\pard\pardeftab720\partightenfactor0
\cf9 \cb3 def\cf4  \cf12 replace_empty_with_none\cf4 (\cf13 tokens_list\cf4 )\cf6 :\cf4 \cb1 \
\pard\pardeftab720\partightenfactor0
\cf4 \cb3     \cf5 return\cf4  \cf7 'None'\cf4  \cf5 if\cf4  \cf12 len\cf6 (\cf4 tokens_list\cf6 )\cf4  == \cf11 0\cf4  \cf5 else\cf4  tokens_list\cb1 \
\
\
\cb3 result\cf6 [\cf7 'Tokenized_Description'\cf6 ]\cf4  = result\cf6 [\cf7 'Tokenized_Description'\cf6 ]\cf4 .apply\cf6 (\cf9 lambda\cf4  x\cf6 :\cf4  replace_empty_with_none\cf6 (\cf4 x\cf6 ))\cf4 \cb1 \
\cb3 \
x=result.values.tolist\cf6 ()\cf4 \cb1 \
\
\cb3 m = \cf12 len\cf6 (\cf4 x\cf6 )\cf4 \cb1 \
\cb3 p_x = \cf6 [\cf4 x\cf6 [:\cf4 i+\cf11 1\cf6 ]\cf4  \cf5 for\cf4  i \cf2 in\cf4  \cf12 range\cf6 (\cf4 m\cf11 -1\cf6 )]\cf4 \cb1 \
\
\pard\pardeftab720\partightenfactor0
\cf8 \cb3 #USING LINEAR LAYERS\cf4 \cb1 \
\
\pard\pardeftab720\partightenfactor0
\cf5 \cb3 import\cf4  torch\cb1 \
\cf5 \cb3 import\cf4  torch.nn \cf5 as\cf4  nn\cb1 \
\cf5 \cb3 import\cf4  torch.nn.functional \cf5 as\cf4  F\cb1 \
\
\pard\pardeftab720\partightenfactor0
\cf9 \cb3 class\cf4  \cf10 twoD_predict\cf4 (\cf10 nn\cf4 .\cf10 Module\cf4 )\cf6 :\cf4 \cb1 \
\
\pard\pardeftab720\partightenfactor0
\cf4 \cb3   \cf9 def\cf4  \cf12 __init__\cf4 (\cf13 self\cf4 )\cf6 :\cf4 \cb1 \
\
\cb3     super\cf6 ()\cf4 .\cf12 __init__\cf6 ()\cf4 \cb1 \
\cb3     \cf13 self\cf4 .f1 = nn.Linear\cf6 (\cf11 3\cf6 ,\cf11 1000\cf6 )\cf4 \cb1 \
\cb3     \cf13 self\cf4 .f2 = nn.Linear\cf6 (\cf11 1000\cf6 ,\cf11 1000\cf6 )\cf4 \cb1 \
\cb3     \cf13 self\cf4 .f3 = nn.Linear\cf6 (\cf11 3\cf6 ,\cf11 1000\cf6 )\cf4 \cb1 \
\cb3     \cf13 self\cf4 .f4 = nn.Linear\cf6 (\cf11 1000\cf6 ,\cf11 1000\cf6 )\cf4 \cb1 \
\cb3     \cf13 self\cf4 .c1 = nn.Linear\cf6 (\cf11 1000\cf6 ,\cf11 1\cf6 )\cf4 \cb1 \
\cb3     \cf13 self\cf4 .c2 = nn.Linear\cf6 (\cf11 1000\cf6 ,\cf11 1\cf6 )\cf4 \cb1 \
\
\cb3   \cf9 def\cf4  \cf12 aggr1\cf4 (\cf13 self\cf4 ,\cf13 n_obvs\cf4 )\cf6 :\cf4 \cb1 \
\
\cb3     var = torch.div\cf6 (\cf4 torch.\cf12 sum\cf6 (\cf4 torch.\cf12 pow\cf6 (\cf4 n_obvs-torch.mean\cf6 (\cf4 n_obvs\cf6 ),\cf11 2\cf6 )),\cf12 len\cf6 (\cf4 n_obvs\cf6 ))\cf4 \cb1 \
\cb3     eps = \cf11 1e03\cf4 \cb1 \
\cb3     centralized_n_obvs = torch.div\cf6 (\cf4 n_obvs-torch.mean\cf6 (\cf4 n_obvs\cf6 ),\cf4 var\cf11 -1\cf4 +eps\cf6 )\cf4 \cb1 \
\cb3     order_means = \cf6 [\cf4 torch.mean\cf6 (\cf4 torch.\cf12 pow\cf6 (\cf4 centralized_n_obvs\cf6 ,\cf4 order\cf6 ))\cf4  \cf5 for\cf4  order \cf2 in\cf4  \cf12 range\cf6 (\cf11 3\cf6 )]\cf4 \cb1 \
\cb3     \cf5 return\cf4  torch.tensor\cf6 (\cf4 order_means\cf6 )\cf4 \cb1 \
\
\cb3   \cf9 def\cf4  \cf12 aggr2\cf4 (\cf13 self\cf4 ,\cf13 c_obvs\cf4 )\cf6 :\cf4 \cb1 \
\
\cb3     n_obvs = c_obvs\cb1 \
\cb3     var = torch.div\cf6 (\cf4 torch.\cf12 sum\cf6 (\cf4 torch.\cf12 pow\cf6 (\cf4 n_obvs-torch.mean\cf6 (\cf4 n_obvs\cf6 ),\cf11 2\cf6 )),\cf12 len\cf6 (\cf4 n_obvs\cf6 ))\cf4 \cb1 \
\cb3     eps = \cf11 1e03\cf4 \cb1 \
\cb3     centralized_n_obvs = torch.div\cf6 (\cf4 n_obvs-torch.mean\cf6 (\cf4 n_obvs\cf6 ),\cf4 var\cf11 -1\cf4 +eps\cf6 )\cf4 \cb1 \
\cb3     order_means = \cf6 [\cf4 torch.mean\cf6 (\cf4 torch.\cf12 pow\cf6 (\cf4 centralized_n_obvs\cf6 ,\cf4 order\cf6 ))\cf4  \cf5 for\cf4  order \cf2 in\cf4  \cf12 range\cf6 (\cf11 3\cf6 )]\cf4 \cb1 \
\cb3     \cf5 return\cf4  torch.tensor\cf6 (\cf4 order_means\cf6 )\cf4 \cb1 \
\
\cb3   \cf9 def\cf4  \cf12 forward\cf4 (\cf13 self\cf4 ,\cf13 x\cf4 )\cf6 :\cf4 \cb1 \
\
\cb3     n_obvs = torch.tensor\cf6 ([\cf10 float\cf6 (\cf4 i\cf6 [\cf11 0\cf6 ])\cf4  \cf5 for\cf4  i \cf2 in\cf4  x\cf6 ])\cf4   \cf8 # Convert to float explicitly\cf4 \cb1 \
\cb3     c_obvs = torch.tensor\cf6 ([\cf4 tokens\cf6 [\cf4 j\cf6 ]\cf4  \cf5 for\cf4  i \cf2 in\cf4  x \cf5 for\cf4  j \cf2 in\cf4  i\cf6 [\cf11 1\cf6 ]],\cf4  dtype=torch.\cf10 float\cf6 )\cf4  \cf8 # Assuming tokens are float values\cf4 \cb1 \
\
\cb3     n_obvs = \cf13 self\cf4 .aggr1\cf6 (\cf4 n_obvs\cf6 )\cf4 \cb1 \
\cb3     l_n = F.leaky_relu\cf6 (\cf13 self\cf4 .f1\cf6 (\cf4 n_obvs\cf6 ))\cf4 \cb1 \
\cb3     l_n = F.leaky_relu\cf6 (\cf13 self\cf4 .f2\cf6 (\cf4 l_n\cf6 ))\cf4 \cb1 \
\cb3     c_n = F.leaky_relu\cf6 (\cf13 self\cf4 .c1\cf6 (\cf4 l_n\cf6 ))\cf4 \cb1 \
\
\cb3     c_obvs = \cf13 self\cf4 .aggr2\cf6 (\cf4 c_obvs\cf6 )\cf4 \cb1 \
\cb3     l_c = F.leaky_relu\cf6 (\cf13 self\cf4 .f3\cf6 (\cf4 c_obvs\cf6 ))\cf4 \cb1 \
\cb3     l_c = F.leaky_relu\cf6 (\cf13 self\cf4 .f4\cf6 (\cf4 l_c\cf6 ))\cf4 \cb1 \
\cb3     c_c = F.leaky_relu\cf6 (\cf13 self\cf4 .c2\cf6 (\cf4 l_c\cf6 ))\cf4 \cb1 \
\
\cb3     \cf5 return\cf4  \cf6 [\cf4 c_n\cf6 ,\cf4 c_c\cf6 ]\cf4 \cb1 \
\
\cb3   \cf9 def\cf4  \cf12 train\cf4 (\cf13 self\cf4 ,\cb1 \
\cb3             \cf13 epochs\cf4  = \cf11 100\cf4 )\cf6 :\cf4 \cb1 \
\
\cb3     optimizer = torch.optim.AdamW\cf6 (\cf13 self\cf4 .parameters\cf6 ())\cf4 \cb1 \
\cb3     n = \cf12 len\cf6 (\cf4 p_x\cf6 )\cf4 \cb1 \
\cb3     \cf5 for\cf4  i \cf2 in\cf4  \cf12 range\cf6 (\cf4 epochs\cf6 ):\cf4 \cb1 \
\cb3       predictions = \cf6 [\cf13 self\cf6 (\cf4 p_x\cf6 [\cf4 j\cf6 ])\cf4  \cf5 for\cf4  j \cf2 in\cf4  \cf12 range\cf6 (\cf4 n\cf6 )]\cf4 \cb1 \
\cb3       gts = torch.tensor\cf6 (\cf4 p_y\cf6 )\cf4 \cb1 \
\cb3       loss = \cf11 0.0\cf4 \cb1 \
\cb3       \cf5 for\cf4  j \cf2 in\cf4  \cf12 range\cf6 (\cf4 n\cf6 ):\cf4 \cb1 \
\cb3         \cf5 for\cf4  l \cf2 in\cf4  \cf12 range\cf6 (\cf11 2\cf6 ):\cf4 \cb1 \
\cb3           loss += torch.\cf12 pow\cf6 (\cf4 predictions\cf6 [\cf4 j\cf6 ][\cf4 l\cf6 ]\cf4 -gts\cf6 [\cf4 j\cf6 ][\cf4 l\cf6 ],\cf11 2\cf6 )\cf4 \cb1 \
\cb3       loss /= n\cb1 \
\cb3       loss.backward\cf6 ()\cf4 \cb1 \
\cb3       print \cf6 (\cf4 loss.item\cf6 ())\cf4 \cb1 \
\cb3       optimizer.step\cf6 ()\cf4 \cb1 \
\cb3       optimizer.zero_grad\cf6 ()\cf4 \cb1 \
\
\cb3 obj1 = twoD_predict\cf6 ()\cf4 \cb1 \
\
\cb3 obj1.train\cf6 ()\cf4 \cb1 \
\
\
\cb3 filename = \cf6 (\cf9 f\cf7 "\cf6 \{\cf4 column_names\cf6 [\cf11 0\cf6 ]\}\cf7 _linearmodel.sav"\cf6 )\cf4 \cb1 \
\cb3 pickle.dump\cf6 (\cf4 obj1\cf6 ,\cf4  \cf12 open\cf6 (\cf4 filename\cf6 ,\cf4  \cf7 'wb'\cf6 ))\cf4 \cb1 \
\
\cb3 x=df_test.values.tolist\cf6 ()\cf4 \cb1 \
\
\cb3 df_test1=df_test\cf6 [[\cf4 column_names\cf6 [\cf11 0\cf6 ],\cf7 'Tokenized_Description'\cf6 ]]\
\
\
\pard\pardeftab720\partightenfactor0
\cf8 # Specify the start_index\cf4 \cb1 \
\pard\pardeftab720\partightenfactor0
\cf4 \cb3 start_index = df_test1.index\cf6 [\cf11 0\cf6 ]\cf4 \cb1 \
\
\pard\pardeftab720\partightenfactor0
\cf8 \cb3 # Initialize a list to store the results\cf4 \cb1 \
\pard\pardeftab720\partightenfactor0
\cf4 \cb3 combined_predictions = \cf6 []\cf4 \cb1 \
\
\pard\pardeftab720\partightenfactor0
\cf8 \cb3 # Loop through different end_index values\cf4 \cb1 \
\pard\pardeftab720\partightenfactor0
\cf5 \cb3 for\cf4  i \cf2 in\cf4  \cf12 range\cf6 (\cf11 100\cf6 ):\cf4   \cf8 # Adjust the number of iterations as needed, taking 100 for now\cf4 \cb1 \
\pard\pardeftab720\partightenfactor0
\cf4 \cb3     end_index = start_index + \cf11 2\cf4  * i \cf8 #2-gap between rows\cf4 \cb1 \
\
\cb3     df_test11 = df_test1.loc\cf6 [\cf4 start_index\cf6 :\cf4 end_index\cf6 ]\cf4 .values.tolist\cf6 ()\cf4 \cb1 \
\cb3     test_predictions = obj1\cf6 (\cf4 df_test11\cf6 )\cf4 \cb1 \
\cb3     combined_predictions.append\cf6 ((\cf4 start_index\cf6 ,\cf4  end_index\cf6 ,\cf4  test_predictions\cf6 ))\cf4 \cb1 \
\
\pard\pardeftab720\partightenfactor0
\cf8 \cb3 # Create a list to store all tensor values\cf4 \cb1 \
\pard\pardeftab720\partightenfactor0
\cf4 \cb3 all_tensor_values_list = \cf6 []\cf4 \cb1 \
\
\pard\pardeftab720\partightenfactor0
\cf8 \cb3 # Process each set of predictions\cf4 \cb1 \
\pard\pardeftab720\partightenfactor0
\cf5 \cb3 for\cf4  _\cf6 ,\cf4  end_index\cf6 ,\cf4  predictions \cf2 in\cf4  combined_predictions\cf6 :\cf4 \cb1 \
\pard\pardeftab720\partightenfactor0
\cf4 \cb3     tensor_values_list = \cf6 [\cf10 float\cf6 (\cf4 tensor_val.item\cf6 ())\cf4  \cf5 for\cf4  tensor_val \cf2 in\cf4  predictions\cf6 ]\cf4 \cb1 \
\cb3     all_tensor_values_list.append\cf6 (\cf4 tensor_values_list\cf6 )\cf4 \cb1 \
\
\cb3 columns = \cf6 [\cf7 'Variable1'\cf6 ,\cf4  \cf7 'Variable2'\cf6 ]\cf4 \cb1 \
\cb3 df_result = pd.DataFrame\cf6 (\cf4 all_tensor_values_list\cf6 ,\cf4  columns=columns\cf6 )\cf4 \cb1 \
\
\cb3 df_result\cf6 [\cf7 'index'\cf6 ]\cf4  = \cf6 [\cf4 end_index + \cf11 1\cf4  \cf5 for\cf4  _\cf6 ,\cf4  end_index\cf6 ,\cf4  _ \cf2 in\cf4  combined_predictions\cf6 ]\cf4 \cb1 \
\
\cb3 df_result\cb1 \
\cb3 df_result.to_csv\cf6 ((\cf9 f\cf7 "\cf6 \{\cf4 column_names\cf6 [\cf11 0\cf6 ]\}\cf7 _linear.csv"\cf6 ))\cf4 \cb1 \
\
\
\cb3 df_test\cf6 [\cf7 'index'\cf6 ]\cf4  = df_test.index\cb1 \
\
\pard\pardeftab720\partightenfactor0
\cf5 \cb3 from\cf4  sklearn.metrics \cf5 import\cf4  classification_report\cf6 ,\cf4  mean_squared_error\cb1 \
\cf5 \cb3 import\cf4  numpy \cf5 as\cf4  np\cb1 \
\
\pard\pardeftab720\partightenfactor0
\cf4 \cb3 merged_df = pd.merge\cf6 (\cf4 df_result\cf6 ,\cf4  df_test\cf6 ,\cf4  on=\cf7 "index"\cf6 )\cf4 \cb1 \
\cb3 print(\cf7 "\cf4 merged_df\cf7 "\cf4 ,merged_df\cb1 )\
\
\cb3 \
classification_report_var1 = classification_report\cf6 (\cf4 merged_df\cf6 [\cf7 'Variable2'\cf6 ]\cf4 .\cf12 round\cf6 (\cf11 0\cf6 ),\cf4  merged_df\cf6 [\cf7 'Average'\cf6 ])\cf4 \cb1 \
\pard\pardeftab720\partightenfactor0
\cf12 \cb3 print\cf6 (\cf7 "classification report"\cf6 ,\cf4 classification_report_var1\cf6 )\cf4 \cb1 \
\
\pard\pardeftab720\partightenfactor0
\cf4 \cb3 rmse_var1 = np.sqrt\cf6 (\cf4 mean_squared_error\cf6 (\cf4 merged_df\cf6 [\cf7 'Variable1'\cf6 ],\cf4  merged_df\cf6 [\cf4 column_names\cf6 [\cf11 0\cf6 ]]))\cf4 \cb1 \
\pard\pardeftab720\partightenfactor0
\cf12 \cb3 print\cf6 (\cf7 "RMSE f"\cf6 \{\cf4 column_names\cf6 [\cf11 0\cf6 ]\}\cf7 ""\cf6 ,\cf4  rmse_var1\cf6 )\cf4 \cb1 \
\
\pard\pardeftab720\partightenfactor0
\cf4 \cb3 rmse_var2 = np.sqrt\cf6 (\cf4 mean_squared_error\cf6 (\cf4 merged_df\cf6 [\cf7 'Variable2'\cf6 ],\cf4  merged_df\cf6 [\cf7 'Average'\cf6 ]))\cf4 \cb1 \
\pard\pardeftab720\partightenfactor0
\cf12 \cb3 print\cf6 (\cf7 "RMSE f"\cf6 \{\cf4 column_names\cf6 [\cf11 1\cf6 ]\}\cf7 ""\cf6 ,\cf4  rmse_var2\cf6 )\cf4 \cb1 \
\
\pard\pardeftab720\partightenfactor0
\cf8 \cb3 #get actual values\cf4 \cb1 \
\pard\pardeftab720\partightenfactor0
\cf4 \cb3 df_test\cf6 [\cf7 'Average'\cf6 ]\cf4  = df_test\cf6 [\cf7 'Tokenized_Description1'\cf6 ]\cf4 .apply\cf6 (\cf4 calculate_average\cf6 )\cf4 \cb1 \
\cb3 df_test\cb1 \
\cb3 df_test1h=df_test\cf6 [[\cf4 column_names\cf6 [\cf11 0\cf6 ],\cf7 'Tokenized_Description'\cf6 ,\cf7 'Average'\cf6 ]]\cf4 \cb1 \
\cb3 df_test1h\cb1 \
\
\
\
\
\
\
}