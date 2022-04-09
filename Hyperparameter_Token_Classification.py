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
    class_weight = [x for x in compute_class_weight("balanced", classes=np.unique(labels), y=labels)]
    print('- auto-computed class weight:', class_weight)
    return class_weight


# A function the creates K separate train/test splits for
# k-fold cross validation
# Passing Parameters:
# dataset: the dataset as a pandas dataframe
# k:  the number of folds
# Return Values:
# A list of k dictionaries
# with each dictionary holding the training
# and test set for that fold.
def make_K_Folds(dataset, k):
    dataset_local = dataset.copy(deep=True)
    folds = []
    splits = []

    for i in range(k, 0, -1):
        split = dataset_local.sample(frac=float(1 / i), random_state=7)
        splits.append(split)
        dataset_local = dataset_local.drop(split.index)

    for split in splits:
        test = split.copy(deep=True)
        train = dataset.drop(split.index)
        folds.append({'train': train, 'test': test})

    return folds


# required for bert to train correctly
# on windows
if __name__ == "__main__":
    # Intitialize the paths to the data folder
    # and the BioBERT-large folder
    cwd = os.getcwd()
    data_folder = Path(cwd + "/Data/")
    model_folder = Path(cwd + "/Models" + "/biobert-large")
    bert_large = model_folder / "biobert-large-cased"
    bert_large_config = model_folder / "biobert-large-cased-config.json"
    bert_large_vocab = model_folder / "biobert-large-cased-vocab.txt"
    output_folder = Path(cwd + "/outputs/")
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)
    # The models to be used in experiments
    model_list = [
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
    tags = ['B-C', 'B-CON', 'B-CS', 'B-EF', 'B-ES', 'B-VC', 'B-VE', 'I-CON', 'I-CS', 'I-ES']
    # The number of epochs
    num_epochs = 5
    # The learning rate
    lr = 2e-5
    # The number of MLP hidden layers
    # set to 0 for linear
    num_mlp = 1

    # For each model and for each tag
    # Create the binary classification dataset
    # and fine tune the model
    for model_name in model_list:

        # Developing a Multi-Class classifier

        # Get the a list of unique labels
        label_list = np.unique(flatten(labels))
        label_list = list(label_list)

        # Convert the data to a dataframe
        data = {'tokens': tokens, 'labels': labels}
        dataset = pd.DataFrame(data=data)

        # get the train/test splits for the 5 folds
        folds = make_K_Folds(dataset, 5)
        # Open the results file for the experiment
        results_file = open(output_folder / (model_name + "_" + "ALL" + "_NER.txt"), "a")
        # write the csv header
        results_file.write("Label;Accuracy;Precision;Recall;F1;Parameters\n")

        # If the model is BioBERT-large
        if model_name == 'biobert-large-cased':
            accuracy = []
            precision = []
            recall = []
            f1 = []
            # For each train/test split in the 5 folds
            for fold in folds:
                train = fold['train']
                test = fold['test']
                class_weights = get_class_weight(flatten(train["labels"]))
                X_train, y_train = train.tokens, train.labels
                # Initialize the BioBERT-large model
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

                # Get the metrics for the fold for all labels
                y_pred = model.predict(test.tokens)
                accuracy.append(accuracy_score(flatten(test.labels), flatten(y_pred)))
                precision.append(
                    precision_score(flatten(test.labels), flatten(y_pred), labels=label_list, average=None))
                recall.append(recall_score(flatten(test.labels), flatten(y_pred), labels=label_list, average=None))
                f1.append(f1_score(flatten(test.labels), flatten(y_pred), labels=label_list, average=None))
            mean_acc = np.mean(accuracy)
            std_acc = np.std(accuracy)
            # format the results and print them to the results file
            if num_mlp != 0:
                parameters = {"epochs": num_epochs, "learning_rate": lr, "num_mlp_layers": num_mlp,
                              "num_mlp_hiddens": 500}
            else:
                parameters = {"epochs": num_epochs, "learning_rate": lr}

            # for each label calculate the average
            # precision, recall and F1-score over the
            # 5 folds along with the standard deviation
            # and print them in the results file
            print("Label |Accuracy        |   Precision      |   Recall        |   F1           | Parameters")
            for i in range(len(label_list)):
                precision_temp = []
                recall_temp = []
                f1_temp = []
                for j in range(len(recall)):
                    precision_temp.append(precision[j][i])
                    recall_temp.append(recall[j][i])
                    f1_temp.append(f1[j][i])
                mean_pre = np.mean(precision_temp)
                std_pre = np.std(precision_temp)
                mean_rec = np.mean(recall_temp)
                std_rec = np.std(recall_temp)
                mean_f1 = np.mean(f1_temp)
                std_f1 = np.std(f1_temp)
                results_file.write(label_list[i] + ";")

                results_file.write(
                    "%0.3f (+/-%0.03f); %0.3f (+/-%0.03f); %0.3f (+/-%0.03f); %0.3f (+/-%0.03f); for %r\n"
                    % (mean_acc, std_acc, mean_pre, std_pre, mean_rec, std_rec
                       , mean_f1, std_f1, parameters))
                print(str(label_list[
                              i]) + ": " + "%0.3f (+/-%0.03f); %0.3f (+/-%0.03f); %0.3f (+/-%0.03f); %0.3f (+/-%0.03f); for %r\n"
                      % (mean_acc, std_acc, mean_pre, std_pre, mean_rec, std_rec
                         , mean_f1, std_f1, parameters))

        # if the model is not biobert-large
        else:
            accuracy = []
            precision = []
            recall = []
            f1 = []
            # For each train/test split in the 5 folds
            for fold in folds:
                train = fold['train']
                test = fold['test']
                class_weights = get_class_weight(flatten(train["labels"]))
                X_train, y_train = train.tokens, train.labels
                # Initialize model
                model = BertTokenClassifier(bert_model=model_name,
                                            max_seq_length=512,
                                            epochs=num_epochs,
                                            gradient_accumulation_steps=2,
                                            learning_rate=lr,
                                            class_weight=class_weights,
                                            train_batch_size=32,
                                            eval_batch_size=32,
                                            validation_fraction=0,
                                            label_list=label_list,
                                            num_mlp_layers=num_mlp,
                                            num_mlp_hiddens=500,
                                            ignore_label=['O'])
                # Train the model
                model.fit(X_train, y_train)
                # Get the metrics for the fold for all labels
                y_pred = model.predict(test.tokens)
                accuracy.append(accuracy_score(flatten(test.labels), flatten(y_pred)))
                precision.append(
                    precision_score(flatten(test.labels), flatten(y_pred), labels=label_list, average=None))
                recall.append(recall_score(flatten(test.labels), flatten(y_pred), labels=label_list, average=None))
                f1.append(f1_score(flatten(test.labels), flatten(y_pred), labels=label_list, average=None))
            mean_acc = np.mean(accuracy)
            std_acc = np.std(accuracy)
            # format the results and print them to the results file
            if num_mlp != 0:
                parameters = {"epochs": num_epochs, "learning_rate": lr, "num_mlp_layers": num_mlp,
                              "num_mlp_hiddens": 500}
            else:
                parameters = {"epochs": num_epochs, "learning_rate": lr}

            # for each label calculate the average
            # precision, recall and F1-score over the
            # 5 folds along with the standard deviation
            # and print them in the results file
            print("Label |Accuracy        |   Precision      |   Recall        |   F1           | Parameters")
            for i in range(len(label_list)):
                precision_temp = []
                recall_temp = []
                f1_temp = []
                for j in range(len(recall)):
                    precision_temp.append(precision[j][i])
                    recall_temp.append(recall[j][i])
                    f1_temp.append(f1[j][i])
                mean_pre = np.mean(precision_temp)
                std_pre = np.std(precision_temp)
                mean_rec = np.mean(recall_temp)
                std_rec = np.std(recall_temp)
                mean_f1 = np.mean(f1_temp)
                std_f1 = np.std(f1_temp)
                results_file.write(label_list[i] + ";")
                results_file.write(
                    "%0.3f (+/-%0.03f); %0.3f (+/-%0.03f); %0.3f (+/-%0.03f); %0.3f (+/-%0.03f); for %r\n"
                    % (mean_acc, std_acc, mean_pre, std_pre, mean_rec, std_rec
                       , mean_f1, std_f1, parameters))
                print(str(label_list[
                              i]) + ": " + "%0.3f (+/-%0.03f); %0.3f (+/-%0.03f); %0.3f (+/-%0.03f); %0.3f (+/-%0.03f); for %r\n"
                      % (mean_acc, std_acc, mean_pre, std_pre, mean_rec, std_rec
                         , mean_f1, std_f1, parameters))

        results_file.close()
