from random import sample
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

from components import composers, envelopes, modifiers, oscillators

st.set_page_config(layout="wide") # make app wide

SR = 44100 # sample rate

# wav helper functions
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

@st.cache(allow_output_mutation=True)
def load_models():
    wv_model, class_labels = load_model_ext('code/models/model2.1mc15-val_acc0.88metadata.h5')
    adsr_model = tf.keras.models.load_model('code/models/model2.2mc22-val_mse0.35')

    return wv_model, class_labels, adsr_model

wv_model, class_labels, adsr_model = load_models()

def make_plots(samples, sample_rate, file_name):
    fig = plt.Figure()
    ax = fig.add_subplot(111)
    ax.specgram(samples, Fs=sample_rate, cmap="plasma", NFFT=256, mode='default')
    canvas = FigureCanvas(fig)
    canvas.print_figure(f'{file_name}_spec_axis.png', bbox_inches='tight')
    ax.axis('off')
    canvas = FigureCanvas(fig)
    canvas.print_figure(f'{file_name}_spec_no_axis.png', bbox_inches='tight')

    fig = plt.Figure()
    ax = fig.add_subplot(111)
    ax.plot(samples)
    canvas = FigureCanvas(fig)
    canvas.print_figure(f'{file_name}_adsr_axis.png', bbox_inches='tight')
    ax.axis('off')
    canvas = FigureCanvas(fig)
    canvas.print_figure(f'{file_name}_adsr_no_axis.png', bbox_inches='tight')

    return None

st.title('Synth Parameter Estimator 🎹')

upload_columns = st.columns((5,5))
with upload_columns[0]:
    st.text('Pitch detection (courtesy of CREPE) is in use.')
    predict_note_octave = st.checkbox('Would you rather select the note and octave?')
    if predict_note_octave == True:
        chosen_note = st.selectbox('Pick note', options=['A', 'A#', 'B', 'C', 'C#', 'D', 'E', 'F', 'F#', 'G', 'G#'])
        chosen_octave = st.number_input('Pick octave', min_value=0, max_value=9)
with upload_columns[1]:
    file = st.file_uploader('Upload an audio file (.wav please)', type=['.wav'], help='audio file to analyze')

# '.mp3', '.m4a', '.flac'

if file is not None:
    sample_rate, samples = wavfile.read(file)
    lib_samp, lib_sample_rate = librosa.load(file)

    if samples.shape[1] > 1:
        samples = librosa.to_mono(lib_samp)
    
    make_plots(samples, sample_rate, 'og')

    og_samp_cols = st.columns((3,3,3))

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

    st.text('Predictions')

    spec_img = load_img('og_spec_no_axis.png')
    adsr_img = load_img('og_adsr_no_axis.png', target_size=(256, 256))
    
    spec_img_array = img_to_array(spec_img)
    spec_img_batch = np.expand_dims(spec_img_array, axis=0)
    st.text((list(json.loads(class_labels).keys())[wv_model.predict(spec_img_batch).argmax(-1)[0]]))

    adsr_img_array = img_to_array(adsr_img)
    adsr_img_batch = np.expand_dims(adsr_img_array, axis=0)
    adsr_estimations = adsr_model.predict(adsr_img_batch/255)[0]

    re_exp = '-?\d*\.\d{3}'

    a_dur = re.match(re_exp, str(adsr_estimations[0]))[0]
    d_dur = re.match(re_exp, str(adsr_estimations[1]))[0]
    s_dur = re.match(re_exp, str(adsr_estimations[3]))[0]
    r_dur = re.match(re_exp, str(adsr_estimations[4]))[0]
    s_lvl = re.match(re_exp, str(adsr_estimations[2]))[0]


    _, pitches, _, _ = crepe.predict(samples[sample_rate*(int(float(a_dur))):(sample_rate*(int(float(a_dur)))+sample_rate)], sr=sample_rate)

    print(librosa.hz_to_note(pitches.mean()))

    st.text(f'''
        Attack Duration: {a_dur}s
        Decay Duration: {d_dur}s
        Sustain Duration: {s_dur}s
        Release Duration: {r_dur}s
        Sustain Level: {float(s_lvl)*100}%
        ''')

    waveform_prediction = (list(json.loads(class_labels).keys())[wv_model.predict(spec_img_batch).argmax(-1)[0]]) 

    if (waveform_prediction == 'sq'):
        waveform = oscillators.SquareOscillator
    elif (waveform_prediction == 'saw'):
        waveform = oscillators.SawtoothOscillator
    if (waveform_prediction == 'sine'):
        waveform = oscillators.SineOscillator
    if (waveform_prediction == 'tri'):
        waveform = oscillators.TriangleOscillator

    a_dur = float(a_dur)
    d_dur = float(d_dur)
    s_lvl = float(s_lvl)
    s_dur = float(s_dur)
    r_dur = float(r_dur)
    
    osc = oscillators.ModulatedOscillator(
        waveform(freq=librosa.note_to_hz(
            chosen_note+str(int(chosen_octave)))
                if predict_note_octave == True
                else pitches.mean()    
            , amp=1),
        envelopes.ADSREnvelope(a_dur, d_dur, s_lvl, s_dur, r_dur),
        amp_mod= lambda init_amp, env: env * init_amp
    )

    wav = get_val(iter(osc), 44100 * 10)

    wave_to_file(wav, fname="estimation.wav")

    est_file = wavfile.read('estimation.wav')

    make_plots(samples=est_file[1], sample_rate=est_file[0], file_name='est')

    est_cols = st.columns((3, 3, 3))

    with est_cols[0]:
        st.subheader('Sample Estimation: Spectrogram')
        st.image(load_img('est_spec_axis.png'), caption='Spectrogram')
    with est_cols[1]:
        st.subheader('Sample Estimation: Amplitude Plot')
        st.image(load_img('est_adsr_axis.png'), caption='Amplitude')
    with est_cols[2]:
        st.subheader('Sample Estimation')
        st.audio(open('estimation.wav', 'rb'), format="audio/wav", start_time=0)

    # oscillators.ModulatedOscillator

    # librosa.load(file)

