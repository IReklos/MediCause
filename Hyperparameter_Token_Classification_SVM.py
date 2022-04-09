from pathlib import Path
import os
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score
import fasttext.util
from sklearn.svm import SVC
from sklearn.utils import compute_class_weight

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

# A function the creates K separate train/test splits for
# k-fold cross validation
# Passing Parameters:
# dataset: the dataset as a pandas dataframe
# k:  the number of folds
# Return Values:
# A list of k dictionaries
# with each dictionary holding the training
# and test set for that fold.
def make_K_Folds(dataset,k):
    dataset_local = dataset.copy(deep=True)
    folds = []
    splits = []

    for i in range(k,0,-1):
        split = dataset_local.sample(frac=float(1/i), random_state=7)
        splits.append(split)
        dataset_local = dataset_local.drop(split.index)

    for split in splits:
        test = split.copy(deep=True)
        train = dataset.drop(split.index)
        folds.append({'train':train,'test':test})

    return folds



cwd = os.getcwd()
# Intitialize the paths to the data folder
# and the outputs folder
data_folder = Path(cwd + "/Data/")
output_folder = Path(cwd + "/outputs/")
if not os.path.exists(output_folder):
    os.mkdir(output_folder)

# Read the data
f = open(data_folder / "EntityRecognition.txt", "r")
data = f.readlines()
tokens = []
labels = []
for i in range(0, len(data) - 1, 2):
    tokens.append(data[i].strip().split(" "))
    labels.append(data[i + 1].strip().split(" "))
f.close()

# Convert the data to a dictionary
data = {'tokens': tokens, 'labels': labels}

# convert the tokens and labels to 1-d lists
tokens = flatten(tokens)
labels = flatten(labels)
# get ehe unique labels
label_list = np.unique(labels)
label_list = list(label_list)

# load the fastText model
fasttext.util.download_model('en', if_exists='ignore')
ft = fasttext.load_model('cc.en.300.bin')
print("Loaded FastText model.")

# convert the data to a dataframe
dataset = pd.DataFrame(data=data)

# get the train/test splits for the 5 folds
folds = make_K_Folds(dataset, 5)

# The c parameters to test
C = [5]#0.5,1,2,5,10
# the kernals to test
kernels = ['rbf']# , 'linear'
# Open the results file for the experiment
results_file = open(output_folder / ("SVM" + "_"+"ALL"  + "_NER.txt"), "a")
# write the csv headers
results_file.write("Label;Accuracy;Precision;Recall;F1;Parameters\n")

# for each C value and kernel combination
for c_param in C:
    for kernel in kernels:

        accuracy = []
        precision = []
        recall = []
        f1 = []
        # For each train/test split in the 5 folds
        for fold in folds:
            train = fold['train']
            test = fold['test']
            # flatten the 2-D lists
            X_train, y_train = flatten(train.tokens), flatten(train.labels)
            X_test, y_test = flatten(test.tokens), flatten(test.labels)
            # vectorize the words
            X_train = [ft.get_word_vector(x).tolist() for x in X_train]
            X_test = [ft.get_word_vector(x).tolist() for x in X_test]

            # Initialize the SVM model
            model = SVC(C=c_param, kernel=kernel,class_weight='balanced')
            # Fit the model to the train data
            model.fit(X_train, y_train)

            # Get the metrics for the fold for all labels
            y_pred = model.predict(X_test)
            accuracy.append(accuracy_score(y_test,y_pred))
            precision.append(precision_score(y_test,y_pred, labels=label_list, average=None))
            recall.append(recall_score(y_test,y_pred,labels=label_list, average=None))
            f1.append(f1_score(y_test,y_pred, labels=label_list, average=None))
        mean_acc = np.mean(accuracy)
        std_acc = np.std(accuracy)
        # format the results and print them to the results file
        parameters = {"C": c_param, "kernel": kernel}

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
            results_file.write("%0.3f (+/-%0.03f); %0.3f (+/-%0.03f); %0.3f (+/-%0.03f); %0.3f (+/-%0.03f); for %r\n"
                               % (mean_acc, std_acc, mean_pre, std_pre, mean_rec, std_rec
                                  , mean_f1, std_f1, parameters))
            print(str(label_list[
                          i]) + ": " + "%0.3f (+/-%0.03f); %0.3f (+/-%0.03f); %0.3f (+/-%0.03f); %0.3f (+/-%0.03f); for %r\n"
                  % (mean_acc, std_acc, mean_pre, std_pre, mean_rec, std_rec
                     , mean_f1, std_f1, parameters))
results_file.close()

