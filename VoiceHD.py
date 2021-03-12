# Basic implementation of VoiceHD without retraining

import matplotlib.pyplot as plt
import multiprocessing
import numpy as np
import sys
import argparse
from scipy.spatial import distance as dst
from sklearn.decomposition import PCA as sklearnPCA
import pandas as pd 
import seaborn as sn
import math, random
from statistics import mean

parser = argparse.ArgumentParser()
parser.add_argument("--plot", dest="plot", action = "store_true")
parser.add_argument('--m', type=int, default=10)
parser.add_argument('--dimension', type=int, default=10000)
parser.add_argument('--retrain', type=int, default=0)
parser.set_defaults(debug=False, plot=False, randomize=False, average=False, dual=False)
args = parser.parse_args()

N = 617              # number of features
D = args.dimension    # hypervector dimensions
M = args.m                 # number of levels

class Record_Based:

    def __init__(self):
        # Create ID vectors
        self.ID = []
        for _ in range(N):
            self.ID.append(np.random.randint(2, size=D))

        # Create level vectors
        self.L = []
        self.L.append(np.random.randint(2, size=D))
        flip = D // (M - 1)
        rand_order = np.random.permutation(range(D))
        count = 0
        for i in range(1, M):
            self.L.append(np.copy(self.L[i-1]))
            for _ in range(flip):
                # rand_order prevents repeating flipped indices
                self.L[i][rand_order[count]] = not self.L[i][rand_order[count]]
                count += 1
        
    def encode(self, inp):
        S = [0] * D
        thresh = N / 2
       
        for i in range(N):
            S = np.add(S, np.bitwise_xor(self.ID[i], self.L[ inp[i] ]))
        return [int(x // thresh) for x in S]    

def init():
    # Bins used to discretize data into M levels    
    bins = np.linspace(-1, 1, num=M, endpoint=False)
    # Load train and test data into (26,0) arrays by class 
    train = [ [] for _ in range(26) ]
    with open('Datasets/isolet1+2+3+4.data', 'r') as f:
        for line in f:
            currentline = list(map(float, line.split(',')))
            train[int(currentline[-1]) - 1].append(currentline[:-1])
    train = bin_data(train, bins)        
    test = [ [] for _ in range(26) ]
    with open('Datasets/isolet5.data', 'r') as f:
        for line in f:
            currentline = list(map(float, line.split(',')))
            test[int(currentline[-1]) - 1].append(currentline[:-1])
    test = bin_data(test, bins)        
    return train, test        


# Round real valued input data to number of network states
def bin_data(data, bins):
    result = [ [] for _ in range(26) ]
    for letter in range(26):
        for line in range(len(data[letter])):
            result[letter].append([x - 1 for x in np.digitize(data[letter][line], bins, right=False)])
    return result    

def parallel_train(train, model, letter, lo):
    my_letter = np.array([0 for _ in range(D)])
    if lo:
        for i in range(120):
            my_letter += np.reshape(model.encode(train[i]), D)
    else:
        for i in range(120, len(train)):
            my_letter += np.reshape(model.encode(train[i]), D)
    letter[:] = np.add(letter, my_letter)        

def parallel_test(test, model, correct_letter, letters, lo, incorrect, prediction):
    offset = 0 if lo else 30
    for i in range(30):
        if i+offset == len(test):
            prediction[i+offset] = correct_letter
            break
        test_letter = np.reshape(model.encode(test[i+offset]), D)
        min_distance = dst.hamming(letters[0], test_letter)
        min_distance_letter = 0
        for j in range(1, 26):
            distance = dst.hamming(letters[j], test_letter)
            if distance < min_distance:
                min_distance = distance
                min_distance_letter = j
        prediction[i+offset] = min_distance_letter        
        if correct_letter != min_distance_letter:
            with incorrect.get_lock():
                incorrect.value += 1


def test_letters(letters, test, model):
    threads = 52
    incorrect = multiprocessing.Value('i', 0)
    prediction = [multiprocessing.Array('i', range(60)) for _ in range(26)]
    print('Testing')
    jobs = []
    for i in range(threads):
        letter = i // 2 
        lo = True if i % 2 == 0 else False
        p = multiprocessing.Process(target=parallel_test,
                args=(test[letter], model, letter, letters, lo, incorrect, 
                    prediction[letter]))
        jobs.append(p)
        p.start()
    for j in jobs:
        j.join()
    return incorrect.value, prediction    


def train_letters(train, model, retrain):
    threads = 52
    letters = [multiprocessing.Array('i', range(D)) for _ in range(26)]

    print('Training')
    for i in range(26):
        letters[i][:] = list(map(lambda x: 0, letters[i]))
    jobs = []
    for i in range(threads):
        letter = i // 2
        lo = True if i % 2 == 0 else False

        p = multiprocessing.Process(target=parallel_train,
                args=(train[letter], model, letters[letter], lo))
        jobs.append(p)
        p.start()
    for j in jobs:
        j.join()

    if not retrain:
        # Threshold class vectors to # number of states
        for i in range(26):
            # Two training data are missing
            instances = 238 if i == 5 else 240
            letters[i] = list(map(lambda x: round(x/instances), letters[i])) 
    return letters

def plot_letters(letters):        
    pca = sklearnPCA(n_components=2)
    transformed = pd.DataFrame(pca.fit_transform(letters))
    letter_list = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm',
            'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
    letter_dict = {'a':0, 'b':1, 'c':2, 'd':3, 'e':4, 'f':5, 'g':6, 'h':7, 'i':8,
            'j':9, 'k':10, 'l':11, 'm':12, 'n':13, 'o':14, 'p':15, 'q':16, 'r':17,
            's':18, 't':19, 'u':20, 'v':21, 'w':22, 'x':23, 'y':24, 'z':25}
    for letter in letter_list:
        x = transformed[0][letter_dict[letter]]
        y = transformed[1][letter_dict[letter]]
        plt.scatter(x, y, marker='x', color='red')
        plt.text(x+.03, y+.03, letter, fontsize=10)
    plt.show(block=False)    

def plot_confusion(prediction):
    plt.figure(2)
    act = np.zeros(1560, dtype=int)
    for i in range(1, 26):
        offset = i * 60  
        for j in range(60):
            act[offset+j] = i
    letter_list = list('abcdefghijklmnopqrstuvwxyz')
    correction = np.arange(26)
    confusion_data = {'actual': np.append(act, correction),
                      'predicted': np.append(np.reshape(prediction, 1560), correction) }
    df = pd.DataFrame(confusion_data, columns=['actual', 'predicted'])
    confusion_matrix = (pd.crosstab(df['actual'], df['predicted'],
                        rownames=['Actual'], colnames=['Predicted'])) #, margins=True))
    sn.heatmap(confusion_matrix, annot=True, yticklabels=letter_list,
            xticklabels=letter_list)
    plt.show(block=False)

def main():
    plot = args.plot
    retrain = args.retrain
    model = Record_Based()

    print('Hypervector dimensions: %i' %(D))
    train, test = init()

    letters = train_letters(train, model, retrain)
    for i in range(20):
        print(letters[0][i], end=' ')
    print()
    sys.exit()

    incorrect, prediction = test_letters(letters, test, model)        
    correct = (1 - (incorrect / 1559)) * 100
    print('Number incorrect %i' %(incorrect))
    print('Correct: %.3f %%' %(correct))

    if plot:
        # Use principle component analysis to project class vectors to two dimensions 
        plot_letters(letters)
        # Plot confusion matrix of results
        plot_confusion(prediction)
        input('Press <ENTER> to continue.')

if __name__ == '__main__':
    main()
