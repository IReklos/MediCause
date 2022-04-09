This project requires a cuda enabled GPU to run correctly

Install pytorch using this command
```
pip3 install torch==1.10.1+cu113 torchvision==0.11.2+cu113 torchaudio===0.10.1+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
```

Then clone this repo and run
```
pip install -r requirements.txt 
```

The Following Python scripts were developed for this project:

•	CohenKappa.py: Calculates the Cohen Kappa statistic between the two annotators. Requires the files Annotator1data.txt and Annotator2data.txt to be inside the Data directory.

•	Dataset.py: Pre-processes the biocausal and the Detecting Causal Language use in Science datasets and saves the resulting dataframes as pickles. Requires the Causaly_small.csv and the pubmed_causal_language_use.csv files to be in the Data directory. After processing the data, the scripts saves the results in the biocausal_data.pickle and data_causal_science.pickle and the total_data.pickle files in the Data directory.

•	Hyperparameter_Causal_detection.py: Develops BERT models for each hyperparameter combination, evaluates their performance and outputs the results in csv format in a .txt file which is named using the convention “model name_dataset_causal”, which is placed in the Outputs directory. Requires the total_data.pickle file to be in the Data directory and the Directory named biobert-large, which contains the BioBERT-large model, to be in the Models directory. 

•	Hyperparameter_Causal_detection_SVM.py: Develops SVM models for each hyperparameter combination, evaluates their performance and outputs the results in csv format in a .txt file named SVM_total_data_causal.txt which is placed in the outputs directory. Requires the total_data.pickle file and the stopwords.txt file to be in the Data directory.

•	Hyperparameter_Token_Classification.py: Develops BERT models for each hyperparameter combination, evaluates their performance and outputs the results in csv format in a .txt file which is named using the convention “model name_label_NER”, which is placed in the Outputs directory. Requires the EntityRecognition.txt file to be in the Data directory and the directory named biobert-large, which contains the BioBERT-large model, to be in the Models directory.

•	Hyperparameter_Token_Classification_SVM.py: Develops SVM models for each hyperparameter combination, evaluates their performance and outputs the results in csv format in a .txt file named SVM_ALL_NER.txt which is placed in the outputs directory. Requires the EntityRecognition.txt file to be in the Data directory.

•	demo.py: An interactive demonstration where the user is asked to input a sentence and the script detects if it contains a causal relation and annotates the entities involved in the relation. Requires the Models directory to contain the Causal and Entity directories, each of whom must contain the bert-causal and bert_large-er models respectively.

The models can be downloaded from this link: https://drive.google.com/file/d/15v8xbPo2BBhTxXr-iiKz4dRLz1d4rKTe/view?usp=sharing

After downloading, unzip the file so that the Models directory is in the same directory as the python scripts

NOTE: The parameters in the Hyperparameter_Causal_detection.py and 	Hyperparameter_Token_Classification.py are meant to fully utilize 24GB of VRAM.
If your GPU has less memory try reducing the max_seq_length and the train_batch_size parameters as well as increasing the gradient_accumulation_steps.
