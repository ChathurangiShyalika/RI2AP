# RI2AP

This repository contains implementation of methods experimented and introduced in the paper titled "Robust and Interpretable 2D Anomaly Prediction in Rocket Assembly Pipelines".

**Baseline models**
This folder includes the baseline models developed.

Three baseline models:
1. LSTM
	* To run py .Baseline Models/LSTM_Model.py
2. Transformer
	* To run py .Baseline Models/Transformer_Model.py
3. TimeGPT
	* To run py .Baseline Models/TimeGPT_Model.py

**Proposed Method**
This folder includes the model for the proposed approach.
* RI2AP 
* Included in the RI2AP Model folder
	* To run py .Baseline Models/RI2AP_Model.py

**Processed Dataset**
* This folder contains the processed dataset which was created from the Future Factories(FF) Dataset (https://www.kaggle.com/datasets/ramyharik/ff-2023-12-12-analog-dataset)

**Combining Predictions**
* This folder contains below files used in adopting combining rules:
1. Noisy-MAX_combine.py 
	* To run py .Combining predictions/Noisy-MAX_combine.py
2. Noisy-OR_combine.py 
	* To run py .Combining predictions/Noisy-OR_combine.py
3. Noisy-OR_and_Noisy-MAX_final_result.py 
	* To run py .Combining predictions/Noisy-OR_and_Noisy-MAX_final_result.py

