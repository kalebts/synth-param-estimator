# Estimating Synthesizer Parameters From Audio Files with Neural Networks

![Common waves](https://en.wikipedia.org/wiki/File:Waveforms.svg)

Synthesizers, or synths, are music instruments that create sounds using electrical signals. These signals come in many forms, but some commonly used ones include the sine wave, sawtooth wave, and a square wave. Each of these waves have a distinct sound that differentiate them from each other and, depending on the synthesizer, can be combined in different ways to make new sounds.

Synthesizers have had a long history being developed as hardware instruments, but now they are available to be used through software, meaning they are more accessible. Synthesizers are a commonly used instrument, due to their versatility and portability. With the advent of increased accesibility comes more music makers/enthusiasts who are curious about the creation of sounds they may hear elsewhere. One community where discussion about this topic occurs is the Reddit forum, [/r/SynthRecipes](www.reddit.com/r/synthrecipes). On this forum, users come with questions about recreating the synths they hear in music, and other users reply with suggestions or solutions to recreating the synth. While this forum is helpful, in the case that someone well-versed in sound synthesis is not available, or the sound is too complicated, an alternative way may be a better option.

## Objective

The purpose of this project is to use neural networks to estimate the parameters of the oscillators that make up synth sound. This will be done using a recreation of a synthesizer in Python by [18alantom](https://github.com/18alantom/synth). Audio files of a synth playing one note will be input for the model to produce estimations. Spectrograms of the audio files will be used to represent the audio frequencies present in the audio file. These will be used in a Convolutional Neural Network model.

The model will be trained with audio files generated randomly using a Python script and tested with files from the same batch of files.

To gauge how successful the estimations are, I will take into account the waveforms that are predicted and the frequency/phase of the waves. If time permits, I would like to incorporate more complicated audio files for the estimation of Attack Delay Sustain Release (ADSR) and filter parameters.

## Repository Structure

📂 audio_files
    -> 📂 1_basic_waves - .wav/.png/.csv files for first model
    -> 📂 2_basic_waves_adsr - .wav/.png/.csv files for second group of models
    -> 📂 3_mult_waves_detune_adsr - .wav/.png/.csv files for third group of models
    -> 📂 archive - .wav/.png files that were not used
    -> 📂 one_offs - .wav/.csv files made for testing

📂 code - containing Jupyter notebooks where I contructed my models

📂 components - portions of 18alantom's synth repo, some of it altered for my project

📂 files - various individual files
    -> 📄 generating_samples.py - script for creating randomized audio samples to train/test models
    -> 📄 requirements.txt - required packages

📂 models - saved Tensorflow models

📂 synth - 18alantom's original synth repo

📄 streamlit-app.py - Streamlit app

## Essential Libraries

### Handling/Modifying data

* Pandas (working with .csv data)

* Numpy (used throughout different packages)

* Scipy (loading/writing .wav files)

### Modeling/Analysis

* Tensorflow (waveform/parameter estimation)

* Sewar* (spectrogram comparison)

* CREPE* (pitch detection)

### Visualizations

* Matplotlib (spectrogram, amplitude plots)

* librosa (amplitude plots)

### Streamlit

### [Synth Code from 18alantom](https://github.com/18alantom/synth)

* not on Anaconda    



Image Source: Omegatron, *Wikimedia*, [Link](https://commons.wikimedia.org/wiki/File:Waveforms.svg), [Creative Commons License](https://creativecommons.org/licenses/by-sa/3.0/)
