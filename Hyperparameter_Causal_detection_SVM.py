from pathlib import Path
import os
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV
import json

# get current working directory
cwd = os.getcwd()
# create the path for data directory
data_folder = Path(cwd + "/Data/")
# create the path for the outputs
output_folder = Path(cwd + "/outputs/")
if not os.path.exists(output_folder):
    os.mkdir(output_folder)
# read in the set of stopwords
with open(data_folder / "stopwords.txt", 'r') as f:
    stopWordsSet = set(json.loads(f.read()))

# read in the data
dataset = "total_data"
data = pd.read_pickle(data_folder / (dataset + ".pickle"))

# the hyperparameter combinations to try in the experiments
parameters = {'C': [0.25], #, 0.75, 1.25, 1.5, 1.75
              'kernel': ['linear'], #, 'rbf'
              }

# open the results csv file and write the headers
results_file = open(output_folder / ("SVM" + "_" + dataset + "_causal.txt"), "a")
results_file.write("Accuracy;Precision;Recall;F1;AUC;Parameters\n")

# create a Tf-idf vectorizer and fit it on the data
vectorizer = TfidfVectorizer(stop_words=stopWordsSet)
vectorizer.fit(data['Sentence'])
# vectorize the sentences
X_train = vectorizer.transform(data["Sentence"])
y_train = data["labels"]

# intialize the SVM classifier and wrap it in GridSearchCV
# to perform 5-fold cross validation for each combination of hyperparameters
clf = GridSearchCV(SVC(),
                   parameters,
                   cv=5,
                   scoring={'accuracy': make_scorer(accuracy_score),
                            'precision': make_scorer(precision_score, pos_label=1, average='binary'),
                            'recall': make_scorer(recall_score, pos_label=1, average='binary'),
                            'f1': make_scorer(f1_score, pos_label=1, average='binary')},
                   verbose=True, refit=False, n_jobs=2)

clf.fit(X_train, y_train)

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
for mean_acc, std_acc, mean_pre, std_pre, mean_rec, std_rec, mean_f1, std_f1, params in \
        zip(means_acc, stds_acc, means_prec, stds_prec, means_rec, stds_rec, means_f1, stds_f1,
            clf.cv_results_['params']):
    print("%0.3f (+/-%0.03f); %0.3f (+/-%0.03f); %0.3f (+/-%0.03f); %0.3f (+/-%0.03f);  for %r"
          % (mean_acc, std_acc, mean_pre, std_pre, mean_rec, std_rec, mean_f1, std_f1,
             params))
    results_file.write("%0.3f (+/-%0.03f); %0.3f (+/-%0.03f); %0.3f (+/-%0.03f); %0.3f (+/-%0.03f); for %r\n"
                       % (mean_acc, std_acc, mean_pre, std_pre, mean_rec, std_rec, mean_f1, std_f1,
                          params))

results_file.close()
