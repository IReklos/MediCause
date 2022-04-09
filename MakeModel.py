import pandas as pd

import os

import sys

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score


from sklearn.utils import compute_class_weight

sys.path.append("../")
from bert_sklearn import BertTokenClassifier


from pathlib import Path




# A function that flattens a 2-D list
# into a 1-D list. The source of the function
# is: https://github.com/charles9n/bert-sklearn
# Passing Parameters:
# l: a 2-D list
# Return values:
# A 1-D list
def flatten(l):
    return [item for sublist in l for item in sublist]

# A function that computes the class weights
# The source of the function is:
# https://github.com/junwang4/causal-language-use-in-science
# Passing parameters:
# labels: A list of labels
# Return values:
# class_weight: The weights for each class
def get_class_weight(labels):
    class_weight = [x for x in compute_class_weight("balanced", np.unique(labels), labels)]
    print('- auto-computed class weight:', class_weight)
    return class_weight


# required for bert to train correctly
# on windows
if __name__ == "__main__":

    # The number of epochs
    num_epochs = 15
    # The learning rate
    lr = 3e-5
    # The number of MLP hidden layers
    # set to 0 for linear
    num_mlp = 1

    # Intitialize the paths to the data folder
    # and the BioBERT-large folder
    cwd = os.getcwd()
    data_folder = Path(cwd + "/Data/")
    model_folder = Path(cwd + "/Models" +"/biobert-large")
    bert_large = model_folder / "biobert-large-cased"
    bert_large_config = model_folder / "biobert-large-cased-config.json"
    bert_large_vocab = model_folder / "biobert-large-cased-vocab.txt"
    output_folder = Path(cwd + "/outputs/")
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)
    # The models to be used in experiments
    model_list=[
                # 'scibert-scivocab-cased',
                # 'biobert-base-cased',
                # 'biobert-v1.0-pubmed-pmc-base-cased',
                'biobert-large-cased'
                ]

    # Read the data
    f = open(data_folder / "EntityRecognition.txt", "r")
    data = f.readlines()
    tokens = []
    labels = []
    for i in range(0, len(data) - 1, 2):
        tokens.append(data[i].strip().split(" "))
        labels.append(data[i + 1].strip().split(" "))
    f.close()
    # The tags of the data
    tags = ['B-C', 'B-CON', 'B-CS','B-EF', 'B-ES','B-VC', 'B-VE', 'I-CON', 'I-CS', 'I-ES']

    # Get the a list of unique labels
    label_list = np.unique(flatten(labels))
    label_list = list(label_list)

    # Convert the data to a dataframe
    data = {'tokens': tokens, 'labels': labels}
    dataset = pd.DataFrame(data=data)

    class_weights = get_class_weight(flatten(dataset["labels"]))
    X_train, y_train = dataset.tokens, dataset.labels

    model = BertTokenClassifier(
        max_seq_length=512,
        epochs=num_epochs,
        train_batch_size=32,
        gradient_accumulation_steps=8, validation_fraction=0,
        learning_rate=lr,
        bert_model=str(bert_large),
        bert_config_json=str(bert_large_config),
        bert_vocab=str(bert_large_vocab),
        class_weight=class_weights,
        eval_batch_size=32,
        num_mlp_layers=num_mlp,
        num_mlp_hiddens=500,
        label_list=label_list,
        ignore_label=['O']
    )
    # Train the model
    model.fit(X_train, y_train)

    model.save("bert_large-er")