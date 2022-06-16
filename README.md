# Estimating Synthesizer Parameters From Audio Samples with Neural Networks

Synthesizers are music instruments that create sounds by generating electrical signals. These signals can come in many forms, but some commonly used ones include the sine wave, sawtooth wave, and a square wave. Each of these waves have a distinct sound that differentiate them from each other and, depending on the synthesizer, can be combined in different ways to make new sounds.

Synthesizers have had a long history being developed as hardware instruments, but now they are available to be used through entirely through software. Nowadays, synthesizers are commonly used in music creation, due to their versatility.

With the advent of increased accesibility comes more music makers/enthusiasts who are curious about the creation of sounds they may hear elsewhere. One community where discussion about this topic occurs is the Reddit forum, [/r/SynthRecipes](www.reddit.com/r/synthrecipes). On this forum, users come with questions about recreating the synths they hear in music, and other users reply with suggestions or solutions to recreating the synth. While this forum is generally helpful, there is the case that someone well-versed in sound synthesis is not available to respond. In this case, an alternative way may be a better option.

## Problem Statement

*How accurately can convolutional neural networks estimate synthesizer waveforms and the amplitude envelope from audio samples?*

## Methods

To generate the samples I used to train and test my models, I used a synthesizer in Python created by [18alantom](https://github.com/18alantom/synth). The generated samples consist of synths playing single notes for up to ten seconds. In the final set of samples, each sample contained between one and four basic waveforms: sine, square sawtooth, and triangle. 

Spectrograms of the audio files were created to represent each sample's audio frequencies over time. These spectrograms were used to train/test a model to detect one or several basic waveforms. Plots of the samples' volumes over time were used to estimate the attack, decay, sustain, and release durations, as well as the volume of a sustained note.

Several iterations of models were created to work with different groups of data, which became slightly more complicated as the project progressed. The final models were trained and tested with spectrograms and amplitude plots of 2000 audio samples.

## Results

Three convolutional neural network models were used to estimate parameters. One CNN model performed multi-label classification using spectrograms of the samples. This model was trained to find patterns in the frequencies emitted by samples (containing between 1-4 waveforms per sample). Two other CNN models were used to provide estimations for ADSR durations and an approximation of the volume for the sustain portion of the envelope.

Different metrics were used to grade both models, since the spectrogram model was used to solve a multi-label classification problem, and the ADSR model was used to solve a regression problem. The spectrogram CNN model had a recall score of 0.7856, a precision score of around 0.7769, and an AUC score of 0.7734 on the test set. Attaining a balance between recall and precision was my goal, since earlier iterations of my model earned higher recall scores because they were blindly predicting too many waveforms per sample. For my amplitude models, the durations model had a mean absolute error of 0.436 and the sustain level model had mean absolute error of 0.055. Since durations was predicting time in seconds, an average error less than half a second is satisfactory. Also, with a mean absolute error of around 5.5% of the maximum volume on the sustain level model, I found the amplitude estimations to be useful.

## Conclusions



The resulting models provide decent performance in estimating synth parameters. With an AUC score of around 0.77, the multilabel model is showing signs of being able to discriminate between different waveforms. The mean absolute errors of the ADSR models are promising as well for generating accurate estimations.

In its current state, these models could still benefit from further work. This is because there are several constraints that should be mentioned. These models were trained on very basic sounds. Only four waveforms were used, so the multi-label model cannot accurately detect anything else. Also, the models do not take into account any detuning that is occuring to the frequencies of the oscillators. This results in estimations for synth sounds not matching actual samples if they are slightly pitched up or down. Also, only samples with a single channel can be predicted, so in the case that a sample consists of two waveforms where each are played exclusively in each channel, they will be treated as if they were routed together in the same channel.

In the future, I would like to include estimations for the presence and parameters of common effects that are often added to sounds, such as filters and filter modulation, as well as delays, reverb, and panning between left and right channels. Inclusion of these types of effects could help estimations of a wider range of samples, especially samples that are more likely to be found in actual songs. If possible, I would also like to figure out a way to automate sample generation with popular software synths to improve the quality of my training dataset.

## Repository Structure

📂 audio_files/mono

    -> 📂 1_basic_waves - .wav/.png/.csv files for first model

    -> 📂 2_basic_waves_adsr - .wav/.png/.csv files for second group of models

    -> 📂 3_mult_waves_detune_adsr - .wav/.png/.csv files for third group of models

    -> 📂 archive - .wav/.png files that were not used

    -> 📂 one_offs - .wav/.csv files made for testing

📂 code - containing Jupyter notebooks where I contructed my models

📂 components - portions of 18alantom's synth repo, some of it altered for my project

<!-- 📂 files - various individual files -->

📂 models - saved Tensorflow models

📂 .streamlit - streamlit app style code

📂 synth - 18alantom's original synth repo

📄 generating_samples.py - script for creating randomized audio samples to train/test models

📄 requirements.txt - required packages

📄 streamlit-app.py - Streamlit app

## Essential Libraries

### Handling/Modifying/Visualizing Data

* Pandas (working with .csv data)

* Numpy (used throughout different packages)

* Scipy (loading/writing .wav files)

* Matplotlib (spectrogram, amplitude plots)

* Librosa (amplitude plots, audio file modifications)

* Pydub (clipping silence from audio samples)

* Streamlit (Streamlit app)

### Modeling/Analysis

* Tensorflow (waveform/parameter estimation)

* Sewar (spectrogram comparison) [pip install necessary]

* CREPE (pitch detection) [pip install necessary]

### [Synth - 18alantom](https://github.com/18alantom/synth)
