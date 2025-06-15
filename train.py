#Importing Libraries
import tensorflow
import numpy as np
import pandas as pd
from collections import Counter
import random
import IPython
from IPython.display import Image, Audio
import music21
from music21 import *
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import tensorflow.keras.backend as K
from tensorflow.keras.optimizers import Adamax
import seaborn as sns

import matplotlib.patches as mpatches
import os
import sys
import warnings
warnings.filterwarnings("ignore")
warnings.simplefilter("ignore")
np.random.seed(42)



# function
def extract_notes(file):
    notes = []
    pick = None
    for j in file:
        songs = instrument.partitionByInstrument(j)
        for part in songs.parts:
            pick = part.recurse()
            for element in pick:
                if isinstance(element, note.Note):
                    notes.append(str(element.pitch))
                elif isinstance(element, chord.Chord):
                    notes.append(".".join(str(n) for n in element.normalOrder))

    return notes


def chords_n_notes(Snippet):
    Melody = []
    offset = 0  # Incremental
    for i in Snippet:
        # If it is chord
        if ("." in i or i.isdigit()):
            chord_notes = i.split(".")  # Separating the notes in chord
            notes = []
            for j in chord_notes:
                inst_note = int(j)
                note_snip = note.Note(inst_note)
                notes.append(note_snip)
            chord_snip = chord.Chord(notes)
            chord_snip.offset = offset
            Melody.append(chord_snip)
        # pattern is a note
        else:
            note_snip = note.Note(i)
            note_snip.offset = offset
            Melody.append(note_snip)
        # increase offset each iteration so that notes do not stack
        offset += 1
    Melody_midi = stream.Stream(Melody)
    return Melody_midi

def train(model_name):
    # Loading the list of chopin's midi files as stream
    filepath = "data/classical-music-midi/"+ model_name + "/"
    # Getting midi files
    all_midis = []
    for i in os.listdir(filepath):
        if i.endswith(".mid"):
            tr = filepath + i
            midi = converter.parse(tr)
            all_midis.append(midi)
    # Getting the list of notes as Corpus
    Corpus = extract_notes(all_midis)
    # Storing all the unique characters present in my corpus to bult a mapping dic.
    symb = sorted(list(set(Corpus)))

    L_corpus = len(Corpus)  # length of corpus
    L_symb = len(symb)  # length of total unique characters

    # Building dictionary to access the vocabulary from indices and vice versa
    mapping = dict((c, i) for i, c in enumerate(symb))
    reverse_mapping = dict((i, c) for i, c in enumerate(symb))

    # Splitting the Corpus in equal length of strings and output target
    length = 40
    features = []
    targets = []
    for i in range(0, L_corpus - length, 1):
        feature = Corpus[i:i + length]
        target = Corpus[i + length]
        features.append([mapping[j] for j in feature])
        targets.append(mapping[target])

    L_datapoints = len(targets)
    print("Total number of sequences in the Corpus:", L_datapoints)
    # reshape X and normalize
    X = (np.reshape(features, (L_datapoints, length, 1))) / float(L_symb)
    # one hot encode the output variable
    y = tensorflow.keras.utils.to_categorical(targets)
    # Taking out a subset of data to be used as seed
    X_train, X_seed, y_train, y_seed = train_test_split(X, y, test_size=0.2, random_state=42)
    # Initialising the Model
    model = Sequential()
    # Adding layers
    model.add(LSTM(512, input_shape=(X.shape[1], X.shape[2]), return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(256))
    model.add(Dense(256))
    model.add(Dropout(0.2))
    model.add(Dense(y.shape[1], activation='softmax'))
    # Compiling the model for training
    opt = Adamax(learning_rate=0.0001)
    model.compile(loss='categorical_crossentropy', optimizer=opt)

    print("******************Model Summary*****************")
    # Model's Summary
    print(model.summary())
    # Training the Model
    history = model.fit(X_train, y_train, batch_size=256, epochs=100)
    name = "models/" + model_name + ".h5"
    model.save(name)
    import matplotlib.pyplot as plt

    # Assuming 'history' contains the training history
    loss = history.history['loss']

    # Plotting the loss curve
    plt.plot(range(1, len(loss) + 1), loss)
    plt.title('Model Loss during Training')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')

    # Save the plot as an image file
    plt.savefig("output/"+model_name+ "training_loss.png")

    # Show the plot
    plt.show()



if __name__ == "__main__":
    # 指定父文件夹路径
    parent_folder = 'data/classical-music-midi'

    # 获取所有子文件夹的名字
    subfolders = [f.name for f in os.scandir(parent_folder) if f.is_dir()]

    # 输出子文件夹名字
    print(subfolders)
    for model_name in subfolders:
        train(model_name)






