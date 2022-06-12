import pandas as pd
import streamlit as st
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from tensorflow.keras.preprocessing.image import img_to_array, load_img, array_to_img, ImageDataGenerator
import numpy as np
import librosa
from scipy.io import wavfile
import h5py
import json
import re
from librosa import display
import math
import crepe
from pydub import AudioSegment
from pydub.silence import detect_nonsilent
from sewar import uqi

from components import composers, envelopes, modifiers, oscillators

st.set_page_config(layout="wide") # make app wide

SR = 44100 # sample rate

# wav helper functions from synth code
to_16 = lambda wav, amp: np.int16(wav * amp * (2**15 - 1))

def get_val(osc, sample_rate=SR):
    return [next(osc) for i in range(sample_rate)]

def wave_to_file(wav, wav2=None, fname="temp.wav", amp=0.1):
    wav = np.array(wav)
    wav = to_16(wav, amp)
    if wav2 != None:
        wav2 = np.array(wav2)
        wav2 = to_16(wav2, amp)
        wav = np.stack([wav, wav2]).T
    
    wavfile.write(fname, SR, wav)

# helper function for loading models with class labels
# code from https://stackoverflow.com/questions/44310448/attaching-class-labels-to-a-keras-model
def load_model_ext(filepath, custom_objects=None):
    model = tf.keras.models.load_model(filepath, custom_objects=None)
    f = h5py.File(filepath, mode='r')
    meta_data = None
    if 'my_meta_data' in f.attrs:
        meta_data = f.attrs.get('my_meta_data')
    f.close()
    return model, meta_data

# loading my models and saving them for quicker streamlit
@st.cache(allow_output_mutation=True)
def load_models():
    '''
        Loads models for streamlit app. @st.cache allows models to be saved after
        starting the app, and avoid being reloaded when the app logs a change.
    '''
    class_labels = None
    # wv_model, class_labels = load_model_ext('models/model3.1mc15-val_acc0.88metadata.h5')
    wv_model = tf.keras.models.load_model('models/model3.1mc15-val_recall0.81')
    adsr_model_durations = tf.keras.models.load_model('models/model3.2_durationsmc31-val_mse0.50')
    adsr_model_s_lvl = tf.keras.models.load_model('models/model3.2_s_lvlmc39-val_mae0.06')

    return wv_model, class_labels, adsr_model_durations, adsr_model_s_lvl

wv_model, class_labels, adsr_model_durations, adsr_model_s_lvl = load_models()

def make_plots(samples, sample_rate, file_name):
    '''
    This function creates spectrogram and amplitude plots for a given array of samples from a
    .wav file. Plots with and without axis are made for display and predicting results, respectively.

    samples: numpy array of samples from wav file
    sample_rate: sample rate for samples
    file_name: identifier for produced plots
    '''

    # spectrogram
    fig = plt.Figure()
    ax = fig.add_subplot(111)
    ax.specgram(samples, Fs=sample_rate, cmap="plasma", NFFT=256, mode='default')
    ax.set_xticks([0, 2, 4, 6, 8, 10])
    canvas = FigureCanvas(fig)
    canvas.print_figure(f'{file_name}_spec_axis.png', bbox_inches='tight')
    ax.axis('off')
    canvas = FigureCanvas(fig)
    canvas.print_figure(f'{file_name}_spec_no_axis.png', bbox_inches='tight')

    # adsr plot
    fig = plt.Figure()
    ax = fig.add_subplot(111)
    ax.plot(samples)
    ax.set_xticks(range(0, 10*sample_rate, sample_rate))
    canvas = FigureCanvas(fig)
    canvas.print_figure(f'{file_name}_adsr_axis.png', bbox_inches='tight')
    ax.axis('off')
    canvas = FigureCanvas(fig)
    canvas.print_figure(f'{file_name}_adsr_no_axis.png', bbox_inches='tight')

    return None


# Start of app

st.title('Synth Parameter Estimator 🎹')


# columns that hold audio upload widget and note/octave selectors
upload_columns = st.columns((5,5))
with upload_columns[0]:
    st.markdown('Pitch detection (courtesy of [CREPE](https://github.com/marl/crepe)) is in use.')
    predict_note_octave = st.checkbox('Would you like to select the note and octave?')
    if predict_note_octave == True:
        chosen_note = st.selectbox('Pick note', options=['A', 'A#', 'B', 'C', 'C#', 'D', 'E', 'F', 'F#', 'G', 'G#'])
        chosen_octave = st.number_input('Pick octave', min_value=0, max_value=9)
with upload_columns[1]:
    file = st.file_uploader('Upload an audio file (.wav please)', type=['.wav'], help='audio file to analyze')

# '.mp3', '.m4a', '.flac'

# if file is uploaded...
if file is not None:
    # read the file
    sample_rate, samples = wavfile.read(file)
    lib_samp, lib_sample_rate = librosa.load(file, sr=44100, mono=False)
    py_samp = AudioSegment.from_wav(file) # for detecting silence
    print(lib_samp.shape)

    # if the sample is stereo, convert to mono
    if len(samples.shape) != 1:
        if samples.shape[1] > 1:
            samples = librosa.to_mono(lib_samp)
            print(samples.shape)
    
    # making sure sample rate is set to the right value
    sample_rate = lib_sample_rate

    # finding the length of the sample to clip silence out
    non_silent_range = detect_nonsilent(py_samp, silence_thresh=-80)

    ns_start = math.floor(sample_rate*(non_silent_range[0][0]/1000))
    ns_finish = math.ceil(sample_rate*(non_silent_range[0][1]/1000))

    # make plots from the sample
    make_plots(samples[ns_start:ns_finish],
        sample_rate, 'og')

    og_samp_cols = st.columns((3,3,3))

    # display the original sample's spectrogram, adsr plot, and audio player
    with og_samp_cols[0]:
        st.subheader('Original Sample: Spectrogram')
        st.image(load_img('og_spec_axis.png'), caption='Spectrogram')
    with og_samp_cols[1]:
        st.subheader('Original Sample: Amplitude Plot')
        st.image(load_img('og_adsr_axis.png'), caption='Amplitude')
    with og_samp_cols[2]:
        st.subheader('Original Sample')
        st.audio(file, format="audio/wav", start_time=0)

        #display.waveshow(lib_samp)

    # predictions

    # load images
    spec_img = load_img('og_spec_no_axis.png', target_size=(256, 256))
    adsr_img = load_img('og_adsr_no_axis.png', target_size=(256, 256))
    
    # process images
    spec_img_array = img_to_array(spec_img)
    spec_img_batch = np.expand_dims(spec_img_array, axis=0)

    adsr_img_array = img_to_array(adsr_img)
    adsr_img_batch = np.expand_dims(adsr_img_array, axis=0)

    # predict adsr envelope
    adsr_dur_estimations = adsr_model_durations.predict(adsr_img_batch/255)[0]
    asdr_s_lvl_estimation = adsr_model_s_lvl.predict(adsr_img_batch/255)[0]

    print(asdr_s_lvl_estimation[0])

    re_exp = '-?\d*\.\d{0,3}'

    a_dur = re.match(re_exp, str(np.abs(adsr_dur_estimations[0])))[0]
    d_dur = re.match(re_exp, str(np.abs(adsr_dur_estimations[1])))[0]
    s_dur = re.match(re_exp, str(np.abs(adsr_dur_estimations[2])))[0]
    r_dur = re.match(re_exp, str(np.abs(adsr_dur_estimations[3])))[0]

    # absolute value of sustain level because predictions can be negative
    s_lvl = re.match(re_exp, str(np.abs(asdr_s_lvl_estimation[0])))[0]

    # get pitch from original sample with crepe
    _, pitches, _, _ = crepe.predict(samples[sample_rate*(int(float(a_dur))):(sample_rate*(int(float(a_dur)))+sample_rate)], sr=sample_rate)

    # average pitches and get closest note to average fråequency
    pitches_avg = librosa.hz_to_note(pitches.mean())

    # convert adsr predictions to floats
    a_dur = float(a_dur)
    d_dur = float(d_dur)
    s_lvl = float(s_lvl)
    s_dur = float(s_dur)
    r_dur = float(r_dur)

    # waveform prediction
    waveform_predictions = wv_model.predict(spec_img_batch)

    waveadder_list = []  # oscillator objects for waveadder object
    waves_list = [] # strings of wave names to display

    for idx, res in enumerate(waveform_predictions[0]):
        if res > .9: # returns decimals, so getting the strongest results
            if idx == 0:
                o = oscillators.SineOscillator
                waves_list.append('Sine')
            elif idx == 1:
                o = oscillators.SquareOscillator
                waves_list.append('Square')
            elif idx == 2:
                o = oscillators.SawtoothOscillator
                waves_list.append('Sawtooth')
            elif idx == 3:
                o = oscillators.TriangleOscillator
                waves_list.append('Triangle')
            waveadder_list.append(
                oscillators.ModulatedOscillator(
                    o(freq=librosa.note_to_hz(
                        chosen_note+str(int(chosen_octave)))
                        if predict_note_octave == True
                        else pitches.mean(), amp=1),
                    envelopes.ADSREnvelope(attack_duration=a_dur, decay_duration=d_dur, sustain_level=s_lvl,
                        sustain_duration=s_dur, release_duration=r_dur),
                    amp_mod=lambda init_amp, env: env * init_amp
            )
        )
    
    # create predicted synth
    osc = composers.WaveAdder(*waveadder_list)

    # export prediction .wav
    wav = get_val(iter(osc), 44100 * 10)

    wave_to_file(wav, fname="estimation.wav")

    # read prediction .wav
    est_file = wavfile.read('estimation.wav')

    # make plots from prediction .wav
    make_plots(samples=est_file[1], sample_rate=est_file[0], file_name='est')

    est_cols = st.columns((3, 3, 3))

    # display estimations
    with est_cols[0]:
        st.subheader('Sample Estimation: Spectrogram')
        st.image(load_img('est_spec_axis.png'), caption='Spectrogram')
    with est_cols[1]:
        st.subheader('Sample Estimation: Amplitude Plot')
        st.image(load_img('est_adsr_axis.png'), caption='Amplitude')
    with est_cols[2]:
        st.subheader('Sample Estimation')
        st.audio(open('estimation.wav', 'rb'), format="audio/wav", start_time=0)
        
        st.metric('Waveform(s)', ", ".join(waves_list))
        if (predict_note_octave==False):
            st.metric('Detected pitch', pitches_avg)

        # Universal Image Quality Index - How close spectrograms resemble each other
        # https://ieeexplore.ieee.org/document/995823
        st.metric(label='UQI - Spectrograms', value=round(uqi(img_to_array(load_img('og_spec_no_axis.png', target_size=(389, 515, 3))), 
            img_to_array(load_img('est_spec_no_axis.png', target_size=(389, 515, 3)))), 3))

        # ADSR estimations
        st.table(
            pd.DataFrame({
                'Attack Duration': [f'{a_dur}s'],
                'Decay Duration': [f'{d_dur}s'],
                'Sustain Duration': [f'{s_dur}s'],
                'Release Duration': [f'{r_dur}s'],
                'Sustain Level': [f'{float(s_lvl)*100}%']
            }).T
        )

