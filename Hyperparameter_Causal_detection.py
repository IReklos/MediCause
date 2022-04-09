from pathlib import Path
import os
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV

from bert_sklearn import BertClassifier

# required for bert to train correctly
# on windows
if __name__ == "__main__":
    # get current working directory
    cwd = os.getcwd()
    # create the path for data directory
    data_folder = Path(cwd + "/Data/")
    # create the path for the biobert-large model
    model_folder = Path(cwd + "/Models" + "/biobert-large")
    bert_large = model_folder / "biobert-large-cased"
    bert_large_config = model_folder / "biobert-large-cased-config.json"
    bert_large_vocab = model_folder / "biobert-large-cased-vocab.txt"
    # create the path for the outputs
    output_folder = Path(cwd + "/outputs/")
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)

    # The list of models to be used in experiments
    model_list = [
        'bert-base-cased',
        'scibert-scivocab-cased',
        'scibert-basevocab-cased',
        'biobert-base-cased',
        'biobert-v1.0-pubmed-pmc-base-cased',
        'biobert-v1.0-pmc-base-cased',
        'biobert-large-cased'
    ]
    # The list of datasets to use in the experiments
    datasets = ["total_data"]

    # for each model and for every dataset
    for model_name in model_list:
        for dataset in datasets:

            # read the training data
            data = pd.read_pickle(data_folder / (dataset + ".pickle"))

            # the hyperparameters to try in the experiments
            parameters = {'epochs': [3, 5, 10],
                          'learning_rate': [1e-5, 2e-5, 3e-5],
                          'num_mlp_layers': [0, 1],
                          'num_mlp_hiddens': [500]
                          }

            # open the results csv file and write the headers
            results_file = open(output_folder / (model_name + "_" + dataset + "_causal.txt"), "a")
            results_file.write("Accuracy;Precision;Recall;F1;AUC;Parameters\n")

            # if the model is biobert-large
            if model_name == 'biobert-large-cased':

                # intialize the BERT classifier and wrap it in GridSearchCV
                # to perform 5-fold cross validation for each combination of hyperparameters
                clf = GridSearchCV(BertClassifier(max_seq_length=512, train_batch_size=32,
                                                  gradient_accumulation_steps=8, validation_fraction=0,
                                                  random_state=7,
                                                  bert_model=str(bert_large),
                                                  bert_config_json=str(bert_large_config),
                                                  bert_vocab=str(bert_large_vocab),
                                                  ),
                                   parameters,
                                   cv=5,
                                   scoring={'accuracy': make_scorer(accuracy_score),
                                            'precision': make_scorer(precision_score),
                                            'recall': make_scorer(recall_score),
                                            'f1': make_scorer(f1_score)},
                                   verbose=True, refit=False)

                clf.fit(data["Sentence"], data["labels"])

                # get the results of the 5-fold cross validation
                means_acc = clf.cv_results_['mean_test_accuracy']
                stds_acc = clf.cv_results_['std_test_accuracy']
                means_prec = clf.cv_results_['mean_test_precision']
                stds_prec = clf.cv_results_['std_test_precision']
                means_rec = clf.cv_results_['mean_test_recall']
                stds_rec = clf.cv_results_['std_test_recall']
                means_f1 = clf.cv_results_['mean_test_f1']
                stds_f1 = clf.cv_results_['std_test_f1']

                # format the results and print them to the results file and the console
                print("Accuracy        |   Precision      |   Recall        |   F1           | Parameters")
                for mean_acc, std_acc, mean_pre, std_pre, mean_rec, std_rec, mean_f1, std_f1, \
                    params in zip(means_acc, stds_acc, means_prec, stds_prec, means_rec, stds_rec, means_f1, stds_f1,
                                  clf.cv_results_['params']):
                    print("%0.3f (+/-%0.03f); %0.3f (+/-%0.03f); %0.3f (+/-%0.03f); %0.3f (+/-%0.03f); for %r"
                          % (mean_acc, std_acc, mean_pre, std_pre, mean_rec, std_rec, mean_f1, std_f1,
                             params))
                    results_file.write(
                        "%0.3f (+/-%0.03f); %0.3f (+/-%0.03f); %0.3f (+/-%0.03f); %0.3f (+/-%0.03f); for %r\n"
                        % (mean_acc, std_acc, mean_pre, std_pre, mean_rec, std_rec, mean_f1, std_f1,
                           params))

            # if the model is not biobert-large
            else:

                # intialize the BERT classifier and wrap it in GridSearchCV
                # to perform 5-fold cross validation for each combination of hyperparameters
                clf = GridSearchCV(BertClassifier(max_seq_length=512, train_batch_size=32,
                                                  gradient_accumulation_steps=2, validation_fraction=0,
                                                  bert_model=model_name, random_state=42,
                                                  ),
                                   parameters,
                                   cv=5,
                                   scoring={'accuracy': make_scorer(accuracy_score),
                                            'precision': make_scorer(precision_score, pos_label=1, average='binary'),
                                            'recall': make_scorer(recall_score, pos_label=1, average='binary'),
                                            'f1': make_scorer(f1_score, pos_label=1, average='binary')},
                                   verbose=True, refit=False)

                clf.fit(data["Sentence"], data["labels"])

                # get the results of the 5-fold cross validation
                means_acc = clf.cv_results_['mean_test_accuracy']
                stds_acc = clf.cv_results_['std_test_accuracy']
                means_prec = clf.cv_results_['mean_test_precision']
                stds_prec = clf.cv_results_['std_test_precision']
                means_rec = clf.cv_results_['mean_test_recall']
                stds_rec = clf.cv_results_['std_test_recall']
                means_f1 = clf.cv_results_['mean_test_f1']
                stds_f1 = clf.cv_results_['std_test_f1']

                # format the results and print them to the results file and the console
                print("Accuracy        |   Precision      |   Recall        |   F1           | Parameters")
                for mean_acc, std_acc, mean_pre, std_pre, mean_rec, std_rec, mean_f1, std_f1, \
                    params in zip(means_acc, stds_acc, means_prec, stds_prec, means_rec, stds_rec, means_f1, stds_f1,
                                  clf.cv_results_['params']):
                    print("%0.3f (+/-%0.03f); %0.3f (+/-%0.03f); %0.3f (+/-%0.03f); %0.3f (+/-%0.03f); for %r"
                          % (mean_acc, std_acc, mean_pre, std_pre, mean_rec, std_rec, mean_f1, std_f1,
                             params))
                    results_file.write(
                        "%0.3f (+/-%0.03f); %0.3f (+/-%0.03f); %0.3f (+/-%0.03f); %0.3f (+/-%0.03f);  for %r\n"
                        % (mean_acc, std_acc, mean_pre, std_pre, mean_rec, std_rec, mean_f1, std_f1,
                           params))

            results_file.close()
