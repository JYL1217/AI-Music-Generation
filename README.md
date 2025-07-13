# Music Generation with LSTM Neural Networks

This project uses LSTM (Long Short-Term Memory) neural networks to generate classical music by training on MIDI files of compositions from various classical composers.
## Project Structure

`train.py`: Trains LSTM models on classical music MIDI data and saves both the trained models and their training loss plots.

`generate.py`: Loads trained models to generate new music, converts predicted notes into MIDI files, and saves the generated melodies.

`data/classical-music-midi/`: Contains subfolders for each composer, with their respective MIDI files.

`models/`: Stores the trained LSTM models.

`output/`: Stores generated MIDI files and training loss plots.
## Setup

Requirements

Ensure you have the following libraries installed:
pip install tensorflow numpy pandas music21 matplotlib sklearn seaborn
## Data

Place MIDI files for each composer in their respective subfolders under data/classical-music-midi/. 
The folder structure should look like this:
```
data/classical-music-midi/
├── Chopin/
│   ├── piece1.mid
│   ├── piece2.mid
│   └── ...
├── Mozart/
│   ├── piece1.mid
│   ├── piece2.mid
│   └── ...
└── ...
```
## Training

To train models for each composer, run:
`python train.py`

Model Training Process

* `Data preparation`: Extracts notes and chords from MIDI files.

* `Model`: Uses LSTM layers with dropout for regularization.

* `Output`: Trained models are saved in the models/ folder, and training loss plots are saved in output/.

## Music Generation

To generate music using trained models, run:
`python generate.py`

Music Generation Process

* `Model loading`: Loads the LSTM model for each composer.

* `Seed sequence`: Randomly selects a valid sequence of notes from the training data.

* `Prediction`: Generates a sequence of notes using the trained model.

* `Output`: Saves the generated MIDI file to output/.

## Outputs

Generated MIDI files: Stored in `output/` with the format `{composer}_generated_music.mid`.

Training loss plots: Saved as `{composer}training_loss.png`.

## Future Enhancements

Add temperature scaling for more creative music generation.

Implement real-time MIDI playback.

Explore more advanced architectures like Transformer models.
