import sys
import numpy as np
import re
import music as M

def init():
    # Save each note as [starting time, pitch, duration] list
    chorales = []
    with open('Datasets/Chorales/chorales.lisp', 'r') as f:
        for line in f:
            chorale = []
            notes = line.split('((')[1:]
            for note in notes:
                note = re.findall('[0-9]+', note)
                chorale.append(note[:3])
            chorales.append(chorale)
    return chorales

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
    clean_chorales = []
    for chorale in chorales:
        note_length = .125 # Fraction of second per 16th note
        freq = [value[1] for value in chorale] # midi frequency
        freq = list(map(lambda x: round(27.5 * 2 ** ((int(x) - 21)/12), 1),
            freq))
        dur = [int(value[2]) for value in chorale] # duration in seconds
        rests_dur = list(map(lambda x: note_length*x, dur)) 
        st = [int(value[0]) for value in chorale]
        duration = st[0]
        counter = 1
        # Insert rests
        for i in range(1, len(st)):
            if st[i] == duration + dur[i-1]:
                duration += dur[i-1]
                counter += 1
            else:
                rests_dur = rests_dur[:counter] + \
                        [note_length * (st[i] - duration - dur[i-1])] + rests_dur[counter:]
                freq = freq[:counter] + [0] + freq[counter:]
                #print('\tRest     st: %i' %(st[i]))
                duration = st[i]
                counter += 2

        # Save duration, pitch pairs of lists    
        clean_chorales.append([freq, rests_dur])
        for f in freq:
            notes.add(f)
    print(sorted(notes))        
    print(len(sorted(notes)))
    return clean_chorales    


def main():
    chorales = init()
    chorales = clean(chorales)
    #record(chorales[93])
    #print(np.shape(chorales))
    #print(np.shape(chorales[0]))
    #print(chorales[0])


if __name__ == '__main__':
    main()
