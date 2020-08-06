import matplotlib.pyplot as plt
import multiprocessing
import numpy as np
import sys
import argparse
from scipy.spatial import distance as dst
from sklearn.decomposition import PCA as sklearnPCA
import pandas as pd 
import seaborn as sn

parser = argparse.ArgumentParser()
parser.add_argument("--i", type=int, default=20) 
parser.add_argument("--k", type=int, default=2) 
parser.add_argument("--debug", dest="debug", action = "store_true")
parser.add_argument("--plot", dest="plot", action = "store_true")
parser.set_defaults(debug=False, plot=False)
args = parser.parse_args()

K = args.k                # in degree connectivity
N = 617              # number of features
I = args.i              # iterations 
LENGTH = N * (I + 1)       # hypervector dimensions
DEBUG = args.debug

class RBN:
    Pow = 2** np.arange(K)
    Connections = np.apply_along_axis(np.random.permutation, 1, np.tile(range(N), (N,1)))[:, 0:K]
    Functions = np.random.randint(0, 2, (N, 2**K))
    State = np.zeros((I+1, N), dtype=int)

    def encode(self, inp):
        self.State[0] = inp 
        for i in range(I):
            self.State[i+1] = self.Functions[:, np.sum(self.Pow * self.State[i, self.Connections], 1)].diagonal()
        return self.State    

class RBN2: # RBN with average in-degree K
    pass


# Round real valued input data to number of network states
def smooth_data(data, states):
    result = [ [] for _ in range(26) ]
    for letter in range(26):
        for line in range(len(data[letter])):
            # Regular threshold
            for number in range(N):
                data[letter][line][number] = min((data[letter][line][number] +  1)  // (2 / float(states)), states-1)
            result[letter].append(list(map(int, data[letter][line])))    
    return result    

def init():
    # Load train and test data into (26,0) arrays by class 
    train = [ [] for _ in range(26) ]
    with open('Datasets/isolet1+2+3+4.data', 'r') as f:
        for line in f:
            currentline = list(map(float, line.split(',')))
            train[int(currentline[-1]) - 1].append(currentline[:-1])
    train = smooth_data(train, 2)        
    test = [ [] for _ in range(26) ]
    with open('Datasets/isolet5.data', 'r') as f:
        for line in f:
            currentline = list(map(float, line.split(',')))
            test[int(currentline[-1]) - 1].append(currentline[:-1])
    test = smooth_data(test, 2)        
    return train, test        

def parallel_train(train, rbn, letter, lo):
    my_letter = np.array([0 for _ in range(LENGTH)])
    if lo:
        for i in range(120):
            my_letter += np.reshape(rbn.encode(train[i]), LENGTH)
    else:
        for i in range(120, len(train)):
            my_letter += np.reshape(rbn.encode(train[i]), LENGTH)
    letter[:] = np.add(letter, my_letter)        

def parallel_test(test, rbn, correct_letter, letters, lo, incorrect, prediction):
    offset = 0 if lo else 30
    for i in range(30):
        if i+offset == len(test):
            prediction[i+offset] = correct_letter
            break
        test_letter = np.reshape(rbn.encode(test[i+offset]), LENGTH)
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
            if DEBUG:
                print('Incorrect: predicted -> %c,  actual -> %c'
                        %(min_distance_letter+97, correct_letter+97))


def test_letters(letters, test, rbn):
    threads = 52
    incorrect = multiprocessing.Value('i', 0)
    prediction = [multiprocessing.Array('i', range(60)) for _ in range(26)]
    print('Testing')
    jobs = []
    for i in range(threads):
        letter = i // 2 
        lo = True if i % 2 == 0 else False
        p = multiprocessing.Process(target=parallel_test,
                args=(test[letter], rbn, letter, letters, lo, incorrect, 
                    prediction[letter]))
        jobs.append(p)
        p.start()
    for j in jobs:
        j.join()
    return incorrect.value, prediction    


def train_letters(train, rbn):
    threads = 52
    letters = [multiprocessing.Array('i', range(LENGTH)) for _ in range(26)]

    print('Training')
    for i in range(26):
        letters[i][:] = list(map(lambda x: 0, letters[i]))
    jobs = []
    for i in range(threads):
        letter = i // 2
        lo = True if i % 2 == 0 else False

        p = multiprocessing.Process(target=parallel_train,
                args=(train[letter], rbn, letters[letter], lo))
        jobs.append(p)
        p.start()
    for j in jobs:
        j.join()

    if DEBUG:
        for j in range(5):
            for i in range(20):
                print(letters[j][i], end=' ')
            print()    
        print()    

    # Threshold class vectors to binary
    for i in range(26):
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

def main():
    plot = args.plot
    rbn = RBN()
    train, test = init()

    print('Hypervector dimensions: %i' %(LENGTH))
    letters = train_letters(train, rbn)
    if DEBUG:
        for j in range(5):
            for i in range(20):
                print(letters[j][i], end=' ')
            print()    
    incorrect, prediction = test_letters(letters, test, rbn)        
    correct = (1 - (incorrect / 1559)) * 100
    print('Number incorrect %i' %(incorrect))
    print('Correct: %.3f %%' %(correct))

    if DEBUG:
        for i in range(26):
            print('%i --> ' %(i))
            for j in range(60):
                print(prediction[i][j], end=' ')
            print()    

    if plot:
        # Plot example RBN from first input datum
        if DEBUG:
            plt.imshow(rbn.encode(train[0][0]))
            plt.show()
        # Use principle component analysis to project class vectors to two dimensions 
        plot_letters(letters)

        # Plot confusion matrix of results
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
        input('Press <ENTER> to continue.')

if __name__ == '__main__':
    main()
