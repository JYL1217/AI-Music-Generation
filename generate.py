import os
from random import random

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

from train import extract_notes, chords_n_notes

warnings.filterwarnings("ignore")
warnings.simplefilter("ignore")
np.random.seed(42)


def generate_music(model_name, num_notes=100):
    # Load model
    model_path = "models/" + model_name + ".h5"
    model = tensorflow.keras.models.load_model(model_path)
    print(f"Loaded model from {model_path}")

    # Prepare data
    filepath = "data/classical-music-midi/" + model_name + "/"
    all_midis = [converter.parse(filepath + i) for i in os.listdir(filepath) if i.endswith(".mid")]
    Corpus = extract_notes(all_midis)

    # Create mapping relationship
    symb = sorted(list(set(Corpus)))
    mapping = {c: i for i, c in enumerate(symb)}
    reverse_mapping = {i: c for i, c in enumerate(symb)}

    # Filter the seed sequence to ensure that only notes included in the mapping dictionary are included
    valid_seed = [note for note in Corpus if note in mapping]
    if len(valid_seed) < 40:
        print("Not enough valid notes for seed sequence. Exiting.")
        return

    # Randomly select legitimate seed sequences
    start_idx = random.randint(0, len(valid_seed) - 40)
    sequence = valid_seed[start_idx:start_idx + 40]
    input_sequence = [mapping[note] for note in sequence]

    generated_notes = []
    for _ in range(num_notes):
        X_input = np.reshape(input_sequence, (1, len(input_sequence), 1)) / float(len(symb))
        prediction = model.predict(X_input, verbose=0)
        index = np.argmax(prediction)
        result = reverse_mapping.get(index, None)

        if result is None:
            break

        generated_notes.append(result)
        input_sequence.append(index)
        input_sequence = input_sequence[1:]

    # Convert to MIDI
    melody = chords_n_notes(generated_notes)
    output_path = "output/" + model_name + "_generated_music.mid"
    melody.write('midi', fp=output_path)
    print(f"Generated music saved to {output_path}")

    # Play the generated MIDI music
    os.system(f"timidity {output_path}")

if __name__ == "__main__":
    parent_folder = 'data/classical-music-midi'
    subfolders = [f.name for f in os.scandir(parent_folder) if f.is_dir()]

    for model_name in subfolders:
        generate_music(model_name)
