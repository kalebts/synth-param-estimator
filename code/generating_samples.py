import numpy as np
import pandas as pd
from scipy.io import wavfile
from components.composers import Chain
from components.envelopes import ADSREnvelope
import matplotlib.pyplot as plt

from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure

from components import composers, envelopes, modifiers, oscillators
from os import listdir, makedirs, mkdir
from librosa import note_to_hz

SR = 44_100 # sample rate for audio files

dur = 10 # how many seconds each clip is

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

def gen_samps(dur, folder, n_samps=20, stereo=False, asdr=None, filter=None):
    for i in range(1, n_samps+1):
        wav, wav2 = None
        ri1 = np.random.randint(3, 10)
        ri2 = np.random.randint(4, 11)

        print(i, ri1, ri2)

        osc = composers.WaveAdder(
            oscillators.SquareOscillator(5*ri1, amp=0.1*ri1),
            oscillators.TriangleOscillator(4*ri2+ri1, amp=0.1*ri2),
            oscillators.SineOscillator(ri1*4*20/ri2)
        )
        
        wav = get_val(iter(osc), 44100 * dur)

        if (stereo == True):
            osc2 = composers.WaveAdder(
                oscillators.SquareOscillator(27.5, amp=0.1),
                oscillators.TriangleOscillator(55, amp=0.5),
                oscillators.SineOscillator(115),
                oscillators.SquareOscillator(220, amp=0.1),
                oscillators.SineOscillator(440,amp=0.3),
                oscillators.TriangleOscillator(880,amp=0.05),
            )
            
            wav2 = get_val(iter(osc2), 44100 * dur)

        if (wav2 == None):
            wave_to_file(wav, fname=folder+ str(i) + ".wav")

# gen_samps(dur, n_samps=10, 'audio_files/mono/')

def amp_mod(init_amp, env):
    return env * init_amp

def freq_mod(init_freq, env, mod_amt=0.01, sustain_level=0.7):
    return init_freq + ((env - sustain_level) * init_freq * mod_amt)

# for name in ['sine', 'sq', 'tri', 'saw']:
#     mkdir('audio_files/mono/basic_waves_ads/' + name)

if (True): # loop to create all the spectrograms
    folder_name = '/2_basic_waves_adsr/'
    sample_len = 10

    notes = ['C', 'C♯', 'D', 'D♯', 'E', 'F', 'F♯', 'G', 'G♯', 'A', 'A♯', 'B']
    octaves = ['1', '2', '3', '4', '5', '6', '7', '8', '9']

    df = pd.DataFrame(columns=['file_name', 'wave', 'note', 'octave', 'note_octave', 
    'freq', 'attack_duration', 'decay_duration', 'sustain_level', 
    'sustain_duration', 'release_duration'])

    for o in [oscillators.SineOscillator, oscillators.SquareOscillator, 
        oscillators.TriangleOscillator, oscillators.SawtoothOscillator]:
        
        for idx in range(1,126):
            a_dur = np.random.uniform(0, 5) # randomize attack
            d_dur = np.random.uniform(0, 5) # randomize decay
            s_lvl = np.random.uniform(0, 1) # randomize sustain level
            s_dur = np.random.uniform(0, 5)  # randomize sustain
            r_dur = np.random.uniform(0, 5) # randomize release 

            if (a_dur + d_dur + s_dur + r_dur > sample_len): # scale down the durations to fit in 10 second clip
                cut_down = sample_len/(a_dur + d_dur + s_dur + r_dur)
                a_dur *= cut_down
                d_dur *= cut_down
                s_dur *= cut_down
                r_dur *= cut_down

            note = notes[np.random.randint(0, 12)] # randomize note
            octave = octaves[np.random.randint(0, 9)] # randomize octave

            osc = oscillators.ModulatedOscillator(
                o(freq=note_to_hz(note+octave), amp=1),
                envelopes.ADSREnvelope(a_dur, d_dur, s_lvl, s_dur, r_dur),
                amp_mod=amp_mod # needed for adsr
                # freq_mod=lambda env, init_freq: freq_mod(env, init_freq, mod_amt=1, sustain_level=0.2)
            )
                    
            wav = get_val(iter(osc), 44100 * dur)

            if (o == oscillators.SineOscillator):
                file_name = 'sine/'
            elif (o == oscillators.SquareOscillator):
                file_name = 'sq/'
            elif (o == oscillators.TriangleOscillator):
                file_name = 'tri/'
            elif (o == oscillators.SawtoothOscillator):
                file_name = 'saw/'

            file_name += 'adsr_' + note + octave + '_' + str(idx)

            df = pd.concat([df, pd.DataFrame({
                'file_name': [file_name + '.wav'],
                'wave': [file_name.split('_')[0]],
                'note': [note],
                'octave': [octave],
                'note_octave': [note+octave],
                'freq': [note_to_hz(note+octave)],
                'attack_duration': [a_dur],
                'decay_duration': [d_dur],
                'sustain_level': [s_lvl],
                'sustain_duration': [s_dur],
                'release_duration': [r_dur]})], ignore_index=True, axis=0)

            wave_to_file(wav, fname=f"audio_files/mono/{folder_name}/wavs/{file_name}.wav")

            # print({
            #     'file_name': [file_name + '.wav'],
            #     'wave': [file_name.split('_')[0]],
            #     'note': [note],
            #     'octave': [octave],
            #     'note_octave': [note+octave],
            #     'freq': [note_to_hz(note+octave)],
            #     'attack_duration': [a_dur],
            #     'decay_duration': [d_dur],
            #     'sustain_level': [s_lvl],
            #     'sustain_duration': [s_dur],
            #     'release_duration': [r_dur]})
            # print(wav[-44100])
            # wav = to_16(np.array(wav), 0.1)
            # print(len(wav))
            # print(wav[-44100])

            # sample_rate, samples = wavfile.read("audio_files/mono/basic_waves_ads/" + file_name + ".wav")

            # with plt.ioff():
            #     # plt.figure(figsize=(12, 8))
            #     # code from https://stackoverflow.com/questions/44165411/most-efficient-way-to-generate-many-figures-with-matplotlib
            #     fig = plt.Figure()
            #     ax = fig.add_subplot(111)
            #     ax.specgram(wav, Fs=SR, cmap="plasma", NFFT=256, mode='default')
            #     ax.axis('off')
            #     canvas = FigureCanvas(fig)
            #     canvas.print_figure("audio_files/mono/" + folder_name + file_name+ '.png', bbox_inches='tight')


    df.to_csv(f'audio_files/mono/{folder_name}/file_data.csv')
else: # testing with single samples
    adsr = envelopes.ADSREnvelope(attack_duration=1, decay_duration=1, sustain_level=.5,
        sustain_duration=2, release_duration=2)

    osc = oscillators.ModulatedOscillator(
            oscillators.SineOscillator(freq=note_to_hz('A4'), amp=1),
            adsr,
            amp_mod=amp_mod, # needed for adsr
            # freq_mod=lambda env, init_freq: freq_mod(env, init_freq, mod_amt=1, sustain_level=0.2)
        )
    # osc.trigger_release()

    wav = get_val(iter(osc), 44100 * dur)
    # wav = to_16(np.array(wav), 0.1)
    wave_to_file(wav, fname="audio_files/mono/one_offs/test1.wav")