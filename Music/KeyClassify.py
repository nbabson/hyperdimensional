# This version builds key signature class vectors from (# of chorales) - TESTS training
# chorales and performs classification on TESTS chorales 

import sys
import numpy as np
import re
import music as M
from scipy.spatial import distance as dst
import copy

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

def record(chorale, name):
    note_length = .125 # Fraction of second per 16th note
    synth = M.core.Being()
    synth.f_ = chorale[0]
    synth.d_ = chorale[1]
    synth.nu_ = [.15] # vibrato depth
    synth.fv_ = [40] # vibrato frequency
    #synth.render(len(synth.f_), 'chorale.wav')
    synth.render(len(synth.f_), name + '.wav')

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

def encode(chorales, note_dict, length_dict):
    chorale_vectors = []

    for chor in chorales:
        #print(chor)
        chorale_vec = [0] * D
        for i in range(len(chor[0])):
            note = np.bitwise_xor(note_dict[chor[0][i]], length_dict[chor[1][i]])
            for rot in range(i): #i):
                note = rotate(note, D)
            chorale_vec += note

        # Threshold vectors
        instances = len(chor[0])
        chorale_vec = list(map(lambda x: round(x/instances), chorale_vec))

        chorale_vectors.append(chorale_vec)
    return chorale_vectors

def make_key_sig_vectors(chorale_vectors, key_sigs):
    key_sig_vectors = [[0] * D for _ in range(9)]
    keys = [0] * 9
    # Keys data is in range -4 to 4, so add 4 for key list 
    for i in range(len(chorale_vectors) - TESTS):
        keys[key_sigs[i] + 4] += 1     
        key_sig_vectors[key_sigs[i]+4] = list(map(sum, zip(key_sig_vectors[key_sigs[i]+4], 
            chorale_vectors[i])))
    for i in range(9):
        key_sig_vectors[i] = list(map(lambda x: round(x/keys[i]), key_sig_vectors[i]))
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
        if min_distance_key != key_sigs[i] + 4:
            incorrect += 1
    print('Incorrect: %i' %(incorrect))        
    return incorrect

def pick_next_note(short_chorales, short_chorale_vectors, note_dict, length_dict,
        short_length, chorales, chorale_number, chorale_vectors):
    #note_duration = .5
    new_chorale_vector = [0] * D 
    '''
    print('Note Dict')
    print(note_dict)
    for x in note_dict:
        print(note_dict[x])
    print(length_dict)    
    print()
    print(short_chorales[0])
    '''
    prediction, predict_length = 0, 0
    predict2, predict3 = 0, 0
    length2, lenbgth3 = 0, 0 
    new_chorale = [[],[]] #copy.deepcopy(short_chorales[chorale_number])
    new_chorale2 = [[],[]] #copy.deepcopy(short_chorales[chorale_number])
    new_chorale3 = [[],[]] #copy.deepcopy(short_chorales[chorale_number])
    for i in range(10):
        min_distance = 1.
        for n in note_dict:
            for l in length_dict:
                next_note = np.bitwise_xor(note_dict[n], length_dict[l])
                for rot in range(i):   # short_length + i):
                    next_note = rotate(next_note, D)
                '''
                        note = np.bitwise_xor(note_dict[chor[0][i]], rest_dict[chor[1][i]])
                        for rot in range(i): #i):
                            note = rotate(note, D)
                '''
                #distance = dst.hamming(next_note, short_chorale_vectors[chorale_number])
                distance = dst.hamming(next_note, chorale_vectors[chorale_number])
                if min_distance > distance:
                    predict3 = predict2
                    predict2 = prediction
                    prediction = n
                    length3 = length2
                    length2 = predict_length
                    predict_length = l
                    min_distance = distance

            #print(next_note, end='\t')
            #print(n, end='\t')
            #print(dst.hamming(next_note, short_chorale_vectors[0]))
        #print('Actual: %f' %chorales[0][0][short_length + i])
        #print('Prediction: %f' %prediction)
        new_chorale[0].append(prediction)
        #new_chorale[1].append(note_duration)
        new_chorale[1].append(predict_length)
        new_chorale2[0].append(predict2)
        new_chorale2[1].append(length2)
        new_chorale3[0].append(predict3)
        new_chorale3[1].append(length3)
    print(new_chorale)
    record(short_chorales[chorale_number], 'short_chorale')
    record(new_chorale, 'new_chorale')
    record(new_chorale2, 'new_chorale2')
    record(new_chorale3, 'new_chorale3')


def main():
    trials = 20
    incorrect = 0
    short_length = 10
    chorale_num = 42

    # Classify key sinatures 
    for i in range(trials):
        chorales, notes, rest, lengths = init()
        chorales, freq_list, key_sigs = clean(chorales)
        note_dict = dict(zip(freq_list, notes))
        note_dict[0] = rest

        intervals = [1, 1, 1, 1, 2, 2, 4, 4]
        intervals[0] =  NOTE_LENGTH
        for i in range(1, len(intervals)):
            intervals[i] = intervals[i-1] + intervals[i] * NOTE_LENGTH
        #lengths_list = [.125, .25, .375, .5, .75, 1., 1.5, 2]  #, 10]
        length_dict = dict(zip(intervals, lengths))
        # Add orthogonal duration vector for note with duration of 80 16th notes
        length_dict[80 * NOTE_LENGTH] = np.random.randint(2, size=D)

        # Make hypervectors of first 'short_length' notes of each chorale
        #short_chorales = [[notes[:short_length] for notes in chor] for chor in chorales]
        #short_chorale_vectors = encode(short_chorales, note_dict, length_dict)

        #record(chorales[1], 'chorale1')
        #record(short_chorales[0], 'short_chorale')


        chorale_vectors = encode(chorales, note_dict, length_dict)
        key_sig_vectors = make_key_sig_vectors(chorale_vectors, key_sigs)
        incorrect += test_key_sigs(key_sig_vectors, chorale_vectors, key_sigs)
    print('Proportion correct: %.4f' %((TESTS * trials - incorrect) / (TESTS * trials))) 
    print('Random: %.4f' %(1/9))
    #pick_next_note(short_chorales, short_chorale_vectors, note_dict, 
    #        length_dict, short_length, chorales, chorale_num, chorale_vectors)

if __name__ == '__main__':
    main()
