
from pathlib import Path
import os
from sklearn.metrics import cohen_kappa_score


# A function that flattens a 2-D list
# into a 1-D list. The source of the function
# is: https://github.com/charles9n/bert-sklearn
# Passing Parameters:
# l: a 2-D list
# Return values:
# A 1-D list
def flatten(l):
    return [item for sublist in l for item in sublist]


# get current working directory
cwd = os.getcwd()
# Intitialize the paths to the data folder
# and the outputs folder
data_folder = Path(cwd + "/Data/")

# read the sentences annotated by annotator 1
f = open(data_folder / "Annotator1data.txt", "r")
data = f.readlines()
tokens1 = []
labels1 = []
for i in range(0, len(data) - 1, 2):
    tokens1.append(data[i].strip().split(" "))
    labels1.append(data[i + 1].strip().split(" "))
f.close()

# read the sentences annotated by annotator 2
f = open(data_folder / "Annotator2data.txt", "r")
data = f.readlines()
tokens2= []
labels2 = []
for i in range(0, len(data) - 1, 2):
    tokens2.append(data[i].strip().split(" "))
    labels2.append(data[i + 1].strip().split(" "))
f.close()

labels =['B-C', 'B-CON', 'B-CS', 'B-EF', 'B-ES', 'B-VC', 'B-VE', 'I-CON', 'I-CS', 'I-ES','O']
# calculate the Cohen's kappa score
kappa = cohen_kappa_score(flatten(labels1), flatten(labels2), labels= labels)
print("The score is: "+ str(kappa))