import pandas as pd
import os
from pathlib import Path


# get the current working directory
cwd = os.getcwd()
data_folder = Path(cwd + "/Data/")
output_folder = Path(cwd + "/outputs/")


# if the outputs directory does not exist create it
if not os.path.exists(output_folder):
    os.mkdir(output_folder)


# Load biocausal dataset into dataframe
data_biocausal = pd.read_csv(data_folder / "Causaly_small.csv", encoding='Latin-1')
# Clean data by keeping only alpha numeric characters and simple punctuation
data_biocausal['Sentence'] = data_biocausal['Sentence'].str.replace('[^a-zA-Z0-9_\-%,.;:/()]', ' ',
                                                                              regex=True)
# Ensure that all words are separated by a single space
data_biocausal['Sentence'] = data_biocausal['Sentence'].str.split().str.join(" ")
# save to pickle
data_biocausal.to_pickle(data_folder / "biocausal_data.pickle")
print("Saved biocausal data")


# Load the causal language in science dataset into dataframe
data_causal_science = pd.read_csv(data_folder / "pubmed_causal_language_use.csv", encoding='Latin-1')
# Convert labels 1,2,3 to 1
data_causal_science['labels'].where(data_causal_science['labels'] == 0, 1, inplace=True)
# Clean data by keeping only alpha numeric characters and simple punctuation
data_causal_science['Sentence'] = data_causal_science['Sentence'].str.replace('[^a-zA-Z0-9_\-%,.;:/()]', ' ',
                                                                              regex=True)
# Ensure that all words are separated by a single space
data_causal_science['Sentence'] = data_causal_science['Sentence'].str.split().str.join(" ")
# save to pickle
data_causal_science.to_pickle(data_folder / "data_causal_science.pickle")
print("Saved causal science data")

# Append both datasets to one dataframe
total_data = data_biocausal.append(data_causal_science, ignore_index=True)

# print the number of sentences with each label
print("Labels of sentences: ")
print(str(total_data.groupby("labels").count()))

# save data to pickle
total_data.to_pickle(data_folder / "total_data.pickle")
