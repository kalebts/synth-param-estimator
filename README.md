# Estimating Synthesizer Parameters From Audio Samples with Neural Networks

<!-- ![Common waves](https://en.wikipedia.org/wiki/File:Waveforms.svg) -->

Synthesizers, or synths, are music instruments that create sounds using electrical signals. These signals come in many forms, but some commonly used ones include the sine wave, sawtooth wave, and a square wave. Each of these waves have a distinct sound that differentiate them from each other and, depending on the synthesizer, can be combined in different ways to make new sounds.

Synthesizers have had a long history being developed as hardware instruments, but now they are available to be used through software, meaning they are more accessible. Synthesizers are a commonly used instrument, due to their versatility and portability. With the advent of increased accesibility comes more music makers/enthusiasts who are curious about the creation of sounds they may hear elsewhere. One community where discussion about this topic occurs is the Reddit forum, [/r/SynthRecipes](www.reddit.com/r/synthrecipes). On this forum, users come with questions about recreating the synths they hear in music, and other users reply with suggestions or solutions to recreating the synth. While this forum is helpful, in the case that someone well-versed in sound synthesis is not available, or the sound is too complicated, an alternative way may be a better option.

## Problem Statement

*Can convolutional neural networks be used to accurately estimate synthesizer waveforms and the amplitude envelope (Attack, Delay, Sustain, Release) from an audio sample?*

## Methods

This project was done using a synthesizer in Python created by [18alantom](https://github.com/18alantom/synth). Generated data is made up of audio samples of synths playing single notes and chords.

Spectrograms of the audio files were created to represent each sample's audio frequencies over time. These spectrograms were used to train/test a model to detect one or several basic waveforms: sine, square, sawtooth, or triangle. Plots of the samples' volume over time were used to estimate the attack, decay, sustain, and release durations, as well as the volume of a sustained note.

Several iterations of models were created to work with different groups of data, which became slightly more complicated as the project progressed. The final models were trained and tested with spectrograms and amplitude plots of 1500 audio samples. These audio samples contained between 1 and 4 unique waveforms, including the sine, sawtooth, square, and triangle waves.

## Results (subject to change by Monday)

Three convolutional neural network models were used to estimate parameters. One CNN model performed multi-label classification using spectrograms of samples. This model was trained to find patterns in the frequencies emitted by samples (containing between 1-4 waveforms per sample). Two other CNN models were used to provide estimations for ADSR durations and an approximation of the volume for the sustain portion of the envelope.

Different metrics were used to grade both models, since the spectrogram model was used to solve a multi-label classification problem, and the ADSR model was used to solve a regression problem. The spectrogram CNN model had a recall score of 0.7856, a precision score of around 0.7769, and an AUC score of 0.7734 on the test set. Attaining a balance between recall and precision was my goal, since earlier iterations of my model earned higher recall scores because they were blindly predicting too many waveforms per sample. For my amplitude models, the durations model had a mean absolute error of 0.436 and the sustain level model had mean absolute error of 0.055. Since durations was predicting time in seconds, an average error less than half a second is satisfactory. Also, with a mean absolute error of around 5.5% of the maximum volume on the sustain level model, I found the amplitude estimations to be useful.

## Conclusions



The resulting models provide decent performance in estimating synth parameters. With 82.9% of waveforms predicted to be present being correct, a user could get a decent idea of where to start with these results. The ADSR model providing a mean absolute error of around 0.549 is promising as well, since this means, on average, estimates are around .549 seconds off from the true values.



In its current state, it should be referred to as a prototype, rather than a product ready for use. This is because there are several constraints that should be mentioned. These models were trained on very basic sounds. Only four waveforms were used and, even though they are very commonly used, the model cannot accurately detect anything else. Also, the amplitude plots feature generally straight lines, which other digital and hardware synths are not always expected to produce. Also, only samples with a single channel can be predicted, so in the case that a sample consists of two waveforms where each are played exclusively in each channel, they will be treated as if they were routed together in the same channel.

In the future, I would like to include estimations for the presence and parameters of common effects that are often added to sounds, such as filters and filter modulation, as well as delays, reverb, and panning between left and right channels. If possible, I would like to figure out a way to automate sample generation with software synths to improve the quality of my training dataset.

## Repository Structure

📂 audio_files

    -> 📂 1_basic_waves - .wav/.png/.csv files for first model

    -> 📂 2_basic_waves_adsr - .wav/.png/.csv files for second group of models

    -> 📂 3_mult_waves_detune_adsr - .wav/.png/.csv files for third group of models

    -> 📂 archive - .wav/.png files that were not used

    -> 📂 one_offs - .wav/.csv files made for testing

📂 code - containing Jupyter notebooks where I contructed my models

📂 components - portions of 18alantom's synth repo, some of it altered for my project

<!-- 📂 files - various individual files -->

📂 models - saved Tensorflow models

📂 synth - 18alantom's original synth repo

📄 generating_samples.py - script for creating randomized audio samples to train/test models

📄 requirements.txt - required packages

📄 streamlit-app.py - Streamlit app

## Essential Libraries

### Handling/Modifying/Visualizing data

* Pandas (working with .csv data)

* Numpy (used throughout different packages)

* Scipy (loading/writing .wav files)

* Matplotlib (spectrogram, amplitude plots)

* librosa (amplitude plots, audio file modifications)

### Modeling/Analysis

* Tensorflow (waveform/parameter estimation)

* Sewar* (spectrogram comparison)

* CREPE* (pitch detection)

### Streamlit

### [Synth - 18alantom](https://github.com/18alantom/synth)

*not on Anaconda    



Image Source: Omegatron, *Wikimedia*, [Link](https://commons.wikimedia.org/wiki/File:Waveforms.svg), [Creative Commons License](https://creativecommons.org/licenses/by-sa/3.0/)
