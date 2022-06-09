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

from components import composers, envelopes, modifiers, oscillators

st.set_page_config(layout="wide") # make app wide

to_16 = lambda wav, amp: np.int16(wav * amp * (2**15 - 1))

def wave_to_file(wav, wav2=None, fname="temp.wav", amp=0.1):
    wav = np.array(wav)
    wav = to_16(wav, amp)
    if wav2 != None:
        wav2 = np.array(wav2)
        wav2 = to_16(wav2, amp)
        wav = np.stack([wav, wav2]).T
    
    wavfile.write(fname, 44100, wav)

# helper function for loading models with class labels
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

st.title('Synth Parameter Estimator 🎹')

file = st.file_uploader('Upload an audio file (.wav please)', type=['.wav', '.mp3', '.m4a', '.flac'], help='audio file to analyze')

if file is not None:
    sample_rate, samples = wavfile.read(file)
    
    fig = plt.Figure()
    ax = fig.add_subplot(111)
    ax.specgram(samples, Fs=sample_rate, cmap="plasma", NFFT=256, mode='default')
    canvas = FigureCanvas(fig)
    canvas.print_figure(f'spec_axis.png', bbox_inches='tight')
    ax.axis('off')
    canvas = FigureCanvas(fig)
    canvas.print_figure(f'spec_no_axis.png', bbox_inches='tight')

    with plt.ioff():
        fig = plt.Figure()
        ax = fig.add_subplot(111)
        ax.plot(samples)
        canvas = FigureCanvas(fig)
        canvas.print_figure(f'adsr_axis.png', bbox_inches='tight')
        ax.axis('off')
        canvas = FigureCanvas(fig)
        canvas.print_figure(f'adsr_no_axis.png', bbox_inches='tight')

    col1, col2 = st.columns((5,5))

    with col1:
        st.image(load_img('spec_axis.png'), caption='Spectrogram')
    with col2:
        st.image(load_img('adsr_axis.png'), caption='Amplitude')

    st.text('Predictions')

    spec_img = load_img('spec_no_axis.png')
    adsr_img = load_img('adsr_no_axis.png', target_size=(256, 256))
    
    img_array = img_to_array(spec_img)
    img_batch = np.expand_dims(img_array, axis=0)
    st.text((list(json.loads(class_labels).keys())[wv_model.predict(img_batch).argmax(-1)[0]]))

    img_array = img_to_array(adsr_img)
    img_batch = np.expand_dims(img_array, axis=0)
    adsr_estimations = adsr_model.predict(img_batch/255)[0]

    re_exp = '-?\d*\.\d{3}'

    a_dur = re.match(re_exp, str(adsr_estimations[0]))[0]
    d_dur = re.match(re_exp, str(adsr_estimations[1]))[0]
    s_dur = re.match(re_exp, str(adsr_estimations[3]))[0]
    r_dur = re.match(re_exp, str(adsr_estimations[4]))[0]
    s_lvl = re.match(re_exp, str(adsr_estimations[2]))[0]
    
    st.text(f'''
        Attack Duration: {a_dur}s
        Decay Duration: {d_dur}s
        Sustain Duration: {s_dur}s
        Release Duration: {r_dur}s
        Sustain Level: {float(s_lvl)*100}%
        ''')

    if ((list(json.loads(class_labels).keys())[wv_model.predict(img_batch).argmax(-1)[0]]) == 'sq'):
        waveform = oscillators.SawtoothOscillator
        


    # librosa.load(file)