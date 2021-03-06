
import sys
import numpy as np
import re
import music as M
from scipy.spatial import distance as dst
import copy
import os
import multiprocessing
import argparse
import pickle
import mido
import random
from sklearn.decomposition import PCA as sklearnPCA
import pandas as pd 
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('--multi', dest='multi', action='store_true')
parser.add_argument('--combined', dest='combined', action='store_true')
parser.add_argument('--midi', dest='midi', action='store_true')
parser.add_argument('--single', dest='single', action='store_true')
parser.add_argument('--d', type=int, default=10000)
parser.add_argument('--input_set_size', type=int, default=2)
parser.add_argument('--random_note', dest='random_note', action='store_true')
parser.add_argument('--plot', dest='plot', action='store_true')
parser.add_argument('--grouped', dest='grouped', action='store_true')
parser.set_defaults(multi=False, combined=False, midi=False, single=False, random_note=False, plot=False)
args = parser.parse_args()
multi = args.multi
combined = args.combined
midi = args.midi
single = args.single
random_note = args.random_note
input_set_size = args.input_set_size
plot = args.plot
grouped = args.grouped

D = args.d
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


def encodeMidi(song, mid_notes_dict, name):

    length = 400
    mid_song = list()
    for i in range(len(song[0])):
        l1 = [1, mid_notes_dict[song[0][i]], 64, 0]
        l2 = [1, mid_notes_dict[song[0][i]], 0, song[1][i] * length]
        mid_song.append(l1)
        mid_song.append(l2)
    mid_song = np.array([[mid_song]])    

    file = mido.MidiFile()
    for i in range(len(mid_song[0])):
        track = mido.MidiTrack()
        track.append(mido.Message('control_change', channel=0, control=0, value=80, time=0))
        track.append(mido.Message('control_change', channel=0, control=32, value=0, time=0))
        track.append(mido.Message('program_change', channel=0, program=50, time=0))
        for j in range(len(mid_song[0][i])):
            note = mido.Message('note_on',channel=int(mid_song[0][i][j][0]), note=int(mid_song[0][i][j][1]), velocity=int(mid_song[0][i][j][2]), time=int(mid_song[0][i][j][3]))
            track.append(note)
        file.tracks.append(track)
    file.save(name + '.mid')    
    #return file


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

        # Add initial rest
        if st[0] != 0:
            freq = [0] + freq
            rests_dur = [NOTE_LENGTH * st[0]] + rests_dur

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

def encode(chorales, note_dict, length_dict, song_end):
    chorale_vectors = []

    for chor in chorales:
        #print(chor)
        chorale_vec = [0] * D
        for i in range(len(chor[0])):
            note = np.bitwise_xor(note_dict[chor[0][i]], length_dict[chor[1][i]])
            for rot in range(i): 
                note = rotate(note, D)
            # Elementwise addition because note is a numpy array    
            chorale_vec += note
        '''    
        # Put marker at end of song
        for i in range(len(chor[0]) + 1):    
            song_end = rotate(song_end, D)
        chorale_vec += song_end    
        '''
        # Threshold vectors
        instances = len(chor[0]) # +1 # Include end marker
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


def parallel_pick_note(rot, n, l, note, length, new_chorale_vector, chorale_vectors, 
        min_distance, prediction, predict_length, predicted_note, predicted_chorale_num, last_pick, length_dict, used_chorales):
    #print('%f    %f' %(n, l))
    #print(note[n])
    #next_note = np.bitwise_xor(note[n], length[l])
    lengths_list = [.125, .25, .375, .5, .75, 1., 1.5, 2, 10]
    # Have each process check three note lengths
    for i in range(3):
    #next_note = np.bitwise_xor(note, length)
        next_note = np.bitwise_xor(note, length_dict[lengths_list[l*3 + i]])
        for _ in range(rot):   # short_length + i):
            next_note = rotate(next_note, D)
        # Threshold new chorale vector
        thresh_new_chorale = list(map(lambda x: round(x/ (rot+1)), 
            new_chorale_vector + next_note))

        # Compare against each chorale individually
        chorale_num = 0
        for chorale in chorale_vectors:
            # Remove most attractive chorale
            #if chorale_num == 5:
            #    chorale_num += 1
            #    continue
            distance = dst.hamming(thresh_new_chorale, chorale)
            # Remove popular chorale 5
            #if min_distance.value > distance:
            #print('Type: ', end='')
            #print(type(min_distance))
            #print('Value: ', end='')
            #print(min_distance.value)
            #print('    ', end='')
            #print(type(distance))
            #if min_distance > distance:

            # don't let successive notes be from same chorale
            #if min_distance.value > distance and last_pick != chorale_num: 
            if min_distance.value > distance and  chorale_num not in used_chorales: 
                #used_chorales.add(chorale_num)
                #print(used_chorales)
                #print(min_distance.value)
                #with prediction.get_lock():
                prediction.value = n
                #with predict_length.get_lock():    
                predict_length.value = lengths_list[l*3 + i]
                #with min_distance.get_lock():    
                min_distance.value = distance
                #with predicted_note.get_lock():    
                #print('Next note: ', end='')
                #print(next_note)
                #predicted_note.value = next_note
                predicted_note[:] = next_note
                #print('Predicted note: ', end='')
                #print(predicted_note.value)
                #with predicted_chorale_num.get_lock():    
                predicted_chorale_num.value = chorale_num
            chorale_num += 1    

def pick_next_note(short_chorales, short_chorale_vectors, note_dict, length_dict,
        short_length, chorales, chorale_number, chorale_vectors, mid_notes_dict, song_end, composition_num):
    #note_duration = .5

    lengths_list = [.125, .25, .375, .5, .75, 1., 1.5, 2, 10]
    # Add all chorales together to represent complete knowledge space
    if combined:
        all_chorales = np.zeros(D) #, dtype=int)
    instances = len(chorales)
    if combined:
        for i in range(instances):
            all_chorales += chorale_vectors[i]
        all_chorales = list(map(lambda x: round(x/instances), all_chorales))
        predicted_chorale_num = -1
    #print(np.shape(all_chorales))
    #print(all_chorales)
    
    # Build new_chorale_vector
    new_chorale_vector = [0] * D 
    new_chorale_vector = np.array(new_chorale_vector)
    used_chorales = set()

    if grouped:
        print('Combining chorale vectors into groups.')
        grouped_chorales = [np.zeros(D) for _ in range(10)]
        print(np.shape(grouped_chorales))
        sys.exit()

    if multi:
        # Parallelize for 63 cores
        #print(len(note_dict)) # 21
        #print(len(length_dict)) # 9
        prediction = multiprocessing.Value('f', 0)
        predict_length = multiprocessing.Value('f', 0)
        predicted_chorale_num = multiprocessing.Value('i', -1)
        predicted_note = multiprocessing.Array('i', range(D))
        last_pick = -1

        new_chorale = [[],[]] #copy.deepcopy(short_chorales[chorale_number])
        # Random first note
        if random_note:
            first_note = random.choice(list(note_dict.keys()))
            first_length = random.choice(list(length_dict.keys()))
            print('First note: %f' %first_note)
            print('First length: %f' %first_length)
            new_chorale[0].append(first_note)
            new_chorale[1].append(first_length)
            new_chorale_vector += np.array(np.bitwise_xor(note_dict[first_note], length_dict[first_length]))
            start_composing = 1
        else:
            start_composing = 0

        #for i in range(100): # short_length):
        for i in range(start_composing, 100): # Random first note
            min_distance = multiprocessing.Value('f', 1.)
            if i % input_set_size == 0:
                used_chorales = set()
            #for l1 in range(3):
            jobs = []
            for n in note_dict: #range(21):
                #for l2 in range(3):
                for length in range(3):
                    #print(l  + (i*3))
                    #length = lengths_list[l2 + (l1*3)]
                    #args=(n, l+(i*3), note_dict[n], length_dict[l+(i*3)],
                    p = multiprocessing.Process(target=parallel_pick_note,
                            args=(i, n, length, note_dict[n], length_dict[.125], #length],
                                new_chorale_vector, chorale_vectors,
                                min_distance, prediction, predict_length, predicted_note, 
                                predicted_chorale_num, last_pick, length_dict, used_chorales))
                    jobs.append(p)
                    p.start()
            for j in jobs:
                #print(j.pid)
                j.join()

            '''
            # Check for end    
            next_note = song_end
            for _ in range(i):   # short_length + i):
                next_note = rotate(next_note, D)
            thresh_new_chorale = list(map(lambda x: round(x/ (i+1)), 
                new_chorale_vector + next_note))
            chorale_num = 0
            m_distance = 1
            for chorale in chorale_vectors:
                distance = dst.hamming(thresh_new_chorale, chorale)

                if m_distance > distance:
                    m_distance = distance
                    end_chorale = chorale_num
                chorale_num += 1    
            #print(song_end)
            print('Song end min distance: %f' %m_distance)
            #print('Song end distance: %f' %(dst.hamming(thresh_new_chorale, next_note)))
            if m_distance < min_distance.value and end_chorale != last_pick:
                print('Song over after %d notes from chorale num: %d' %(i, end_chorale))
                break
            '''
            # End song if longer than chosen chorale
            if i >= len(chorales[predicted_chorale_num.value][0]):
                print('End song length %d of chorale %d' %(i+1, predicted_chorale_num.value)) 
                break
            #print(predicted_note.value)            
            #print(np.array(predicted_note))            
            new_chorale_vector += np.array(predicted_note)
            #print(new_chorale_vector)
            print('%f         %d' %(min_distance.value, predicted_chorale_num.value))    
            used_chorales.add(predicted_chorale_num.value)
            print(used_chorales)
            new_chorale[0].append(round(prediction.value, 1))
            new_chorale[1].append(predict_length.value)
            last_pick = predicted_chorale_num.value
    else:                    
        prediction, predict_length = 0, 0
        new_chorale = [[],[]] #copy.deepcopy(short_chorales[chorale_number])
        for i in range(30): #short_length):
            min_distance = 1.
            for n in note_dict:
                for l in length_dict:
                    next_note = np.bitwise_xor(note_dict[n], length_dict[l])
                    for rot in range(i):   # short_length + i):
                        next_note = rotate(next_note, D)

                    # Compare against each chorale individually
                    #for chorale in chorale_vectors:
                    #    distance = dst.hamming(next_note, chorale)
                        #distance = dst.hamming(next_note, chorale_vectors[chorale_number])
                        #distance = dst.hamming(new_chorale_vector+next_note, chorale_vectors[chorale_number])

                    # Threshold new chorale vector
                    thresh_new_chorale = list(map(lambda x: round(x/ (i+1)), 
                        new_chorale_vector + next_note))
                    #distance = dst.hamming(next_note, all_chorales)
                    #distance = dst.hamming(new_chorale_vector, all_chorales)

                    # Compare against combined chorales
                    if combined:
                        distance = dst.hamming(thresh_new_chorale, all_chorales)
                        if min_distance > distance:
                            prediction = n
                            predict_length = l
                            min_distance = distance
                            predicted_note = next_note

                    # Compare against single chorale
                    elif single:
                        distance = dst.hamming(thresh_new_chorale, chorale_vectors[chorale_number])
                        #print('         %f' %distance)
                        if min_distance > distance:
                            prediction = n
                            predict_length = l
                            min_distance = distance
                            predicted_note = next_note
                            predicted_chorale_num = chorale_number

                    # Compare against each chorale individually
                    else:
                        chorale_num = 0
                        for chorale in chorale_vectors:
                            distance = dst.hamming(thresh_new_chorale, chorale)

                            if min_distance > distance:
                                prediction = n
                                predict_length = l
                                min_distance = distance
                                predicted_note = next_note
                                predicted_chorale_num = chorale_num
                            chorale_num += 1    

            new_chorale_vector += predicted_note
            print('%f         %d' %(min_distance, predicted_chorale_num))    
            new_chorale[0].append(prediction)
            new_chorale[1].append(predict_length)
    print(new_chorale)

    #print(np.shape(new_chorale_vector))
    #print(new_chorale_vector)

    #record(short_chorales[chorale_number], 'short_chorale')
    if combined:
        if midi:
            encodeMidi(new_chorale, mid_notes_dict, '../../Music/combined_predict')
        else:    
            record(new_chorale, '../../Music/combined_predict')
    else:    
        if midi:
            encodeMidi(new_chorale, mid_notes_dict, '../../Music/parallel_predict')
            #encodeMidi(new_chorale, mid_notes_dict, '../Datasets/ComposedChorales/comp' + str(composition_num))
        else:    
            record(new_chorale, '../../Music/parallel_predict')
            #np.save('../../Music/chorale_array', new_chorale)
    #record(chorales[chorale_number], 'original_chorale')



def plot_chorales(chorales, key_sigs, boring, random_chorales):        
    pca = sklearnPCA(n_components=2)
    transformed = pd.DataFrame(pca.fit_transform(chorales))
    #print(np.shape(transformed))
    #print(transformed[0])
    #letter_list = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm',
    #        'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
    #letter_dict = {'a':0, 'b':1, 'c':2, 'd':3, 'e':4, 'f':5, 'g':6, 'h':7, 'i':8,
    #        'j':9, 'k':10, 'l':11, 'm':12, 'n':13, 'o':14, 'p':15, 'q':16, 'r':17,
    #        's':18, 't':19, 'u':20, 'v':21, 'w':22, 'x':23, 'y':24, 'z':25}
    #for letter in letter_list:
    #    x = transformed[0][letter_dict[letter]]
    #    y = transformed[1][letter_dict[letter]]
    #    plt.scatter(x, y, marker='x', color='red')
    #    plt.text(x+.03, y+.03, letter, fontsize=10)
    for i in range(20): #100):
        x = transformed[0][i]
        y = transformed[1][i]
        plt.scatter(x, y, marker='x', color='red')
        plt.text(x+.03, y+.03, i, fontsize=20) #key_sigs[i], fontsize=20)

    # Add boring chorales    
    transformed = pd.DataFrame(pca.fit_transform(boring))
    for i in range(20):
        x = transformed[0][i]
        y = transformed[1][i]
        plt.scatter(x, y, marker='x', color='blue')
        plt.text(x+.03, y+.03, 'B', fontsize=20)

    # Add random chorales    
    transformed = pd.DataFrame(pca.fit_transform(random_chorales))
    for i in range(20):
        x = transformed[0][i]
        y = transformed[1][i]
        plt.scatter(x, y, marker='x', color='green')
        plt.text(x+.03, y+.03, 'R', fontsize=20)

    plt.show(block=False)    
    input('Press <ENTER> to continue.')
    sys.exit()


def main():
    trials = 20
    incorrect = 0
    short_length = 20
    chorale_num = 5 

    for comp_num in range(1): #100)
        #print('Play Chorale')
        #os.system('paplay short_chorale.wav')
        chorales, notes, rest, lengths = init()
        chorales, freq_list, key_sigs = clean(chorales)
        note_dict = dict(zip(freq_list, notes))
        note_dict[0] = rest

        # Make notes dictionary for midi recording
        mid_notes = []
        for i in range(len(freq_list)):
            mid_notes.append(i+60)
        mid_notes_dict = dict(zip(freq_list, mid_notes))    
        mid_notes_dict[0] = 0

        '''
        # Record midi files of chorales dataset
        chor_num = 0
        for chorale in chorales:
            encodeMidi(chorale, mid_notes_dict, '../Datasets/ChoralesMidi/chorale' + str(chor_num))
            chor_num += 1
        sys.exit()    
        '''
        
        intervals = [1, 1, 1, 1, 2, 2, 4, 4]
        intervals[0] =  NOTE_LENGTH
        for i in range(1, len(intervals)):
            intervals[i] = intervals[i-1] + intervals[i] * NOTE_LENGTH
        #lengths_list = [.125, .25, .375, .5, .75, 1., 1.5, 2]  #, 10]
        length_dict = dict(zip(intervals, lengths))
        # Add orthogonal duration vector for note with duration of 80 16th notes
        length_dict[80 * NOTE_LENGTH] = np.random.randint(2, size=D)
        song_end = np.random.randint(2, size=D)

        # Make hypervectors of first 'short_length' notes of each chorale
        short_chorales = [[notes[:short_length] for notes in chor] for chor in chorales]
        short_chorale_vectors = encode(short_chorales, note_dict, length_dict, song_end)

        #record(chorales[5], 'chorale5')
        #record(short_chorales[0], 'short_chorale')
        #encodeMidi(chorales[10], mid_notes_dict, '../../Music/chorale10')
        #sys.exit()        

        # Generate random and boring (repeated single note) chorales for comparison 
        print('Generate random chorales')
        for i in range(100, 200):
            new_chorale = [[],[]] 
            for _ in range(33):
                new_chorale[0].append(random.choice(list(note_dict.keys())))
                new_chorale[1].append(random.choice(list(length_dict.keys())))
            #print(new_chorale)
            encodeMidi(new_chorale, mid_notes_dict, '../Datasets/RandomChorales/random' + str(i))
        '''    
        for i in range(100):
            new_chorale = [[],[]] 
            random_note = random.choice(list(note_dict.keys()))
            for _ in range(33):
                new_chorale[0].append(random_note)
                new_chorale[1].append(random.choice(list(length_dict.keys())))
            encodeMidi(new_chorale, mid_notes_dict, '../Datasets/BoringChorales/boring' + str(i))
        '''
        sys.exit()    

        if plot:
            # Record random chorale
            random_chorales = []
            for i in range(20):
                new_chorale = [[],[]] 
                for _ in range(30):
                    new_chorale[0].append(random.choice(list(note_dict.keys())))
                    new_chorale[1].append(random.choice(list(length_dict.keys())))
                #print(new_chorale)
                #encodeMidi(new_chorale, mid_notes_dict, '../Datasets/RandomChorales/random' + str(i))
                random_chorales.append(new_chorale)
            random_chorales = encode(random_chorales, note_dict, length_dict, song_end)

            # Record boring single repeated note chorales
            boring = []
            for _ in range(20):
                new_chorale = [[],[]] 
                random_note = random.choice(list(note_dict.keys()))
                for _ in range(35):
                    new_chorale[0].append(random_note)
                    new_chorale[1].append(random.choice(list(length_dict.keys())))
                #encodeMidi(new_chorale, mid_notes_dict, '../../Music/boring1')
                boring.append(new_chorale)
            #new_chorale = [[],[]] 
            #random_note = random.choice(list(note_dict.keys()))
            #random_length = random.choice(list(length_dict.keys()))
            #for _ in range(35):
            #    new_chorale[0].append(random_note)
            #    new_chorale[1].append(random_length)
            #encodeMidi(new_chorale, mid_notes_dict, '../../Music/boring2')
            #boring.append(new_chorale)
            boring = encode(boring, note_dict, length_dict, song_end)

        chorale_vectors = encode(chorales, note_dict, length_dict, song_end)

        if plot:
            plot_chorales(chorale_vectors, key_sigs, boring, random_chorales)

        #key_sig_vectors = make_key_sig_vectors(chorale_vectors, key_sigs)
        #incorrect += test_key_sigs(key_sig_vectors, chorale_vectors, key_sigs)
        #print('Proportion correct: %.4f' %((TESTS * trials - incorrect) / (TESTS * trials))) 
        #print('Random: %.4f' %(1/9))
        pick_next_note(short_chorales, short_chorale_vectors, note_dict, 
                length_dict, short_length, chorales, chorale_num, chorale_vectors, mid_notes_dict, song_end, comp_num)

if __name__ == '__main__':
    main()
