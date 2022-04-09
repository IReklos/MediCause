
import os
import sys
sys.path.append("../")
from bert_sklearn import load_model
from pathlib import Path
import torch

# required for bert to run correctly
# on windows
if __name__ == "__main__":
    cwd = os.getcwd()
    # load the causal sentence detection model from file
    causal_model = load_model(Path(cwd + "/Models" +"/Causal" + "/bert-causal"))
    # load the entity recognition model from file
    er_model = load_model(Path(cwd + "/Models" + "/Entity" + "/bert_large-er"))
    while(True):
        # get input sentence from user
        sentence = input("Please enter a sentence or type Q to exit: ")
        if sentence == 'Q':
            break
        print("Checking if sentence contains causal relation: ")
        # check if it contains a causal relation
        pred = causal_model.predict([sentence])
        # if it does not contain a causal relation
        if pred[0] == 0:
            print("The sentence does not appear to contain a causal relation.")
        # if it contains a causal relation
        else:
            # get the predicted tags for each word
            list_sentence = sentence.split()
            pred_er = er_model.predict([list_sentence])
            # print the words and the tags next to each other
            print("The sentence appears to contain the following relation: ")
            for i in range(len(list_sentence)):
                print(list_sentence[i] + " " + pred_er[0][i])
            


