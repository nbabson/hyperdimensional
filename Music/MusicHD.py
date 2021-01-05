# Classify chorales by key siganature
# Achieves slightly better than twice 
# accuracy over random prediction

import sys
import numpy as np
import re
import music as M
from scipy.spatial import distance as dst

D = 10000
NOTES = 20
NOTE_LENGTH = .125 # Fraction of second per 16th note
TESTS = 20 # Size of test data

def init():
    # Save each note as [starting time, pitch, duration] list
    chorales = []
    with open('../Datasets/Chorales/chorales.lisp', 'r') as f:
        for line in f:
            chorale = []
            notes = line.split('((')[1:]
            for note in notes:
                #note = re.findall('[0-9]+', note)
                note = re.findall('-?\d+', note)

                chorale.append(note[:4])
            chorales.append(chorale)
    Rest = np.random.randint(2, size=D)
    Notes = []
    Notes.append(np.random.randint(2, size=D))
    flip = D // ((NOTES - 1) * 2)
    rand_order = np.random.permutation(range(D))
    count = 0
    for i in range(1, NOTES):
        Notes.append(np.copy(Notes[i-1]))
        for _ in range(flip):
            # rand_order prevents repeating flipped indices
            Notes[i][rand_order[count]] = not Notes[i][rand_order[count]]
            count += 1
    intervals = [1, 1, 1, 1, 2, 2, 4, 4]
    Lengths = []
    # Make note length vectors dictionary 
    Lengths.append(np.random.randint(2, size=D))
    flip = D // ((16 - 1) * 2)
    rand_order = np.random.permutation(range(D))
    count = 0
    for i in range(1, 8):
        Lengths.append(np.copy(Lengths[i-1]))
        for _ in range(flip * intervals[i]):
            # rand_order prevents repeating flipped indices
            Lengths[i][rand_order[count]] = not Lengths[i][rand_order[count]]
            count += 1

    return chorales, Notes, Rest, Lengths

def record(chorale):
    note_length = .125 # Fraction of second per 16th note
    synth = M.core.Being()
    synth.f_ = chorale[0]
    synth.d_ = chorale[1]
    synth.nu_ = [.15] # vibrato depth
    synth.fv_ = [40] # vibrato frequency
    synth.render(len(synth.f_), 'chorale.wav')

def clean(chorales):
    notes = set()
    #lengths = set()
    clean_chorales = []
    key_sigs = []
    for chorale in chorales:
        freq = [value[1] for value in chorale] # midi frequency
        freq = list(map(lambda x: round(27.5 * 2 ** ((int(x) - 21)/12), 1),
            freq))
        dur = [int(value[2]) for value in chorale] # duration in seconds
        rests_dur = list(map(lambda x: NOTE_LENGTH*x, dur)) 
        st = [int(value[0]) for value in chorale]

        key_sigs.append(int(chorale[0][3])) 

        duration = st[0]
        counter = 1
        # Insert rests
        for i in range(1, len(st)):
            if st[i] == duration + dur[i-1]:
                duration += dur[i-1]
                counter += 1
            else:
                rests_dur = rests_dur[:counter] + \
                        [NOTE_LENGTH * (st[i] - duration - dur[i-1])] + rests_dur[counter:]
                freq = freq[:counter] + [0] + freq[counter:]
                #print('\tRest     st: %i' %(st[i]))
                duration = st[i]
                counter += 2

        # Save duration, pitch lists for each chorale   
        clean_chorales.append([freq, rests_dur])
        for f in freq:
            notes.add(f)
        #for d in rests_dur:
        #    lengths.add(d)
    return clean_chorales, sorted(notes)[1:], key_sigs    

def rotate(vec, length):
    #return vec[length-1:] + vec[:length-1]
    return np.concatenate((vec[length-1:], vec[:length-1]))

def encode(chorales, note_dict, rest_dict):
    chorale_vectors = []

    for chor in chorales:
        #print(chor)
        chorale_vec = [0] * D
        for i in range(len(chor[0])):
            note = np.bitwise_xor(note_dict[chor[0][i]], rest_dict[chor[1][i]])
            '''
            print(note_dict[chor[0][i]])
            print(rest_dict[chor[1][i]])
            print(len(note))
            print(note)
            print(i)
            '''
            for rot in range(i): #i):
                #print('Rotate')
                note = rotate(note, D)
                #print(np.shape(note), end='')
                #print(len(note), end='')
                #print(type(note), end='')
                #print(note)
            #print('i = %i\t' %(i), end='')    
            #print(note)
            chorale_vec += note

            #if i == 20:
            #    sys.exit()
        #print(chorale_vec)


        # Threshold vectors
        instances = len(chor[0])
        chorale_vec = list(map(lambda x: round(x/instances), chorale_vec))

        '''
        ones = 0
        zeros = 0
        for i in range(len(chorale_vec)):
            if chorale_vec[i] == 0:
                zeros += 1
            else:
                ones += 1
        print('zeros = %i\tones = %i' %(zeros, ones))       
        '''

        chorale_vectors.append(chorale_vec)
        #print(np.shape(chorale_vectors))
    return chorale_vectors

def make_key_sig_vectors(chorale_vectors, key_sigs):
    key_sig_vectors = [[0] * D for _ in range(9)]
    keys = [0] * 9
    '''
    maximum, minimum = -100, 200
    for k in key_sigs:
        #print(k)
        if k < minimum:
            minimum = k
        if k > maximum:
            maximum = k
    '''        
    # Keys data is in range -4 to 4, so add 4 for key list 
    for i in range(len(chorale_vectors) - TESTS):
        keys[key_sigs[i] + 4] += 1     
        #print(key_sigs[i])
        #key_sig_vectors[key_sigs[i]] += chorale_vectors[i]
        key_sig_vectors[key_sigs[i]+4] = list(map(sum, zip(key_sig_vectors[key_sigs[i]+4], 
            chorale_vectors[i])))
    #print('Max: %i\tMin: %i' %(maximum, minimum))        
    #print(keys)
    # Threshold key sig class vectors    
    #chorale_vec = list(map(lambda x: round(x/instances), chorale_vec))
    '''
    for i in range(9):
        for j in range(20):
            print(key_sig_vectors[i][j], end=' ')
        print()    
    print()    
    '''
    for i in range(9):
        key_sig_vectors[i] = list(map(lambda x: round(x/keys[i]), key_sig_vectors[i]))
    #print(np.shape(key_sig_vectors))
    '''
    for i in range(9):
        for j in range(20):
            print(key_sig_vectors[i][j], end=' ')
        print()    

    print()
    for i in range(20):
        print(chorale_vectors[7][i], end=' ')
    print()    
    '''
    return key_sig_vectors

def test_key_sigs(key_sig_vectors, chorale_vectors, key_sigs):
    incorrect = 0
    for i in range(len(chorale_vectors) - TESTS, len(chorale_vectors)):
        min_distance = dst.hamming(chorale_vectors[i], key_sig_vectors[0])
        min_distance_key = 0
        for j in range(1, 9):
            distance = dst.hamming(chorale_vectors[i], key_sig_vectors[j])
            if distance < min_distance:
                min_distance = distance
                min_distance_key = j
        #print('Prediction: %i\tActual: %i' %(min_distance_key, key_sigs[i] + 4))        
        if min_distance_key != key_sigs[i] + 4:
            incorrect += 1
    print('Incorrect: %i' %(incorrect))        
    return incorrect


def main():
    trials = 20
    incorrect = 0
    for _ in range(trials):
        chorales, notes, rest, lengths = init()
        chorales, freq_list, key_sigs = clean(chorales)
        note_dict = dict(zip(freq_list, notes))
        note_dict[0] = rest

        intervals = [1, 1, 1, 1, 2, 2, 4, 4]
        intervals[0] =  NOTE_LENGTH
        for i in range(1, len(intervals)):
            intervals[i] = intervals[i-1] + intervals[i] * NOTE_LENGTH
        #lengths_list = [.125, .25, .375, .5, .75, 1., 1.5, 2]  #, 10]
        rest_dict = dict(zip(intervals, lengths))
        # Add orthogonal duration vector for note with duration of 80 16th notes
        rest_dict[80 * NOTE_LENGTH] = np.random.randint(2, size=D)

        '''
        for i in range(len(notes)):
            print(dst.hamming(notes[0], notes[i]))
        print()
        print(dst.hamming(np.random.randint(2, size=D), np.random.randint(2, size=D)))
        print(dst.hamming(notes[0], note_dict[0]))
        print(chorales[0])
        print(np.shape(chorales))
        for i in range(len(lengths)):
            print(dst.hamming(lengths[0], lengths[i]))
        print()
        print(chorales[0])
        #print(note_dict)
        #print(np.shape(chorales))
        #print(np.shape(chorales[0]))
        #print(chorales[0])
        print(rest_dict)
        print(dst.hamming(lengths[0], rest_dict[10]))
        '''
        #record(chorales[70])
        chorale_vectors = encode(chorales, note_dict, rest_dict)
        key_sig_vectors = make_key_sig_vectors(chorale_vectors, key_sigs)
        incorrect += test_key_sigs(key_sig_vectors, chorale_vectors, key_sigs)
    print('Proportion correct: %.4f' %((TESTS * trials - incorrect) / (TESTS * trials))) 
    print('Random: %.4f' %(1/9))

if __name__ == '__main__':
    main()
