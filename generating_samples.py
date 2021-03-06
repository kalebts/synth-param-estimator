import random
import numpy as np
import pandas as pd
from scipy.io import wavfile
from components.composers import Chain, WaveAdder
from components.envelopes import ADSREnvelope
import matplotlib.pyplot as plt

from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure

from components import composers, envelopes, modifiers, oscillators
from os import listdir, makedirs, mkdir
from librosa import note_to_hz

SR = 44_100 # sample rate for audio files

dur = 10 # how many seconds each clip is

##########################
#### code from /synth ####
##########################

to_16 = lambda wav, amp: np.int16(wav * amp * (2**15 - 1))

def get_val(osc, sample_rate=SR): # 
    return [next(osc) for i in range(sample_rate)]

def wave_to_file(wav, wav2=None, fname="temp.wav", amp=0.1):
    wav = np.array(wav)
    wav = to_16(wav, amp)
    if wav2 != None:
        wav2 = np.array(wav2)
        wav2 = to_16(wav2, amp)
        wav = np.stack([wav, wav2]).T
    
    wavfile.write(fname, SR, wav)

def amp_mod(init_amp, env):
    return env * init_amp

def freq_mod(init_freq, env, mod_amt=0.01, sustain_level=0.7):
    return init_freq + ((env - sustain_level) * init_freq * mod_amt)

##########################
#### code from /synth ####
##########################

##########################
# loop for creating samples
# please set the part you want to run to True to generate samples
##########################
if (False): # Group 2: Single waves with adsr
    folder_name = '/2_basic_waves_adsr/' # folder_name for exported files
    sample_len = 10 # sample length in seconds

    notes = ['C', 'C♯', 'D', 'D♯', 'E', 'F', 'F♯', 'G', 'G♯', 'A', 'A♯', 'B']
    octaves = ['1', '2', '3', '4', '5', '6', '7', '8', '9']

    # initalizing dataframe
    df = pd.DataFrame(columns=['file_name', 'wave', 'note', 'octave', 'note_octave', 
    'freq', 'attack_duration', 'decay_duration', 'sustain_level', 
    'sustain_duration', 'release_duration'])

    # creating 125 samples for each oscillator
    for o in [oscillators.SineOscillator, oscillators.SquareOscillator, 
        oscillators.TriangleOscillator, oscillators.SawtoothOscillator]:
        
        for idx in range(1,126):
            a_dur = np.random.uniform(0, 5) # randomize attack
            d_dur = np.random.uniform(0, 5) # randomize decay
            s_lvl = np.random.uniform(0, 1) # randomize sustain level
            s_dur = np.random.uniform(0, 5)  # randomize sustain
            r_dur = np.random.uniform(0, 5) # randomize release 

            # scale down the durations to fit in 10 second clip
            if (a_dur + d_dur + s_dur + r_dur > sample_len):
                cut_down = sample_len/(a_dur + d_dur + s_dur + r_dur)
                a_dur *= cut_down
                d_dur *= cut_down
                s_dur *= cut_down
                r_dur *= cut_down

            note = notes[np.random.randint(0, 12)] # randomize note
            octave = octaves[np.random.randint(0, 9)] # randomize octave

            osc = oscillators.ModulatedOscillator( # creating monophonic sample with adsr
                o(freq=note_to_hz(note+octave), amp=1),
                envelopes.ADSREnvelope(a_dur, d_dur, s_lvl, s_dur, r_dur),
                amp_mod=amp_mod # needed for adsr
            )
            
            wav = get_val(iter(osc), 44100 * dur) # wav file

            # placing sample in the right folder
            if (o == oscillators.SineOscillator):
                file_name = 'sine/'
            elif (o == oscillators.SquareOscillator):
                file_name = 'sq/'
            elif (o == oscillators.TriangleOscillator):
                file_name = 'tri/'
            elif (o == oscillators.SawtoothOscillator):
                file_name = 'saw/'

            file_name += 'adsr_' + note + octave + '_' + str(idx)

            # add data to dataframe
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

            # export .wav file
            wave_to_file(wav, fname=f"audio_files/mono/{folder_name}/wavs/{file_name}.wav")

    # export .csv
    df.to_csv(f'audio_files/mono/{folder_name}/file_data.csv') # export dataframe
elif (True): # Group 3: making samples with multiple waves, ADSR, slightly detuned
    folder_name = '3_mult_waves_detune_adsr' # folder name
    sample_len = 10     # seconds of sample

    notes = ['C', 'C♯', 'D', 'D♯', 'E', 'F', 'F♯', 'G', 'G♯', 'A', 'A♯', 'B']
    octaves = ['1', '2', '3', '4', '5', '6', '7', '8', '9']

    # initalizing dataframe
    df = pd.DataFrame(columns=['file_name', 'sine_wave', 'sine_freq', 'square_wave', 'square_freq', 
        'saw_wave', 'saw_freq', 'triangle_wave', 'triangle_freq', 'note', 'octave', 'note_octave', 
        'attack_duration', 'decay_duration', 'sustain_level', 'sustain_duration', 'release_duration'])
        
    starting_range_index = 1
    for idx in range(starting_range_index,starting_range_index + 1500):
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

        # making a row for the dataframe
        osc_info = {
            'file_name': [],
            'sine_wave': [0],
            'sine_freq': [0],
            'square_wave': [0],
            'square_freq': [0], 
            'saw_wave': [0],
            'saw_freq': [0],
            'triangle_wave': [0],
            'triangle_freq': [0], 
            'note': [note],
            'octave': [octave],
            'note_octave': [note+octave],
            'attack_duration': [a_dur],
            'decay_duration': [d_dur],
            'sustain_level': [s_lvl],
            'sustain_duration': [s_dur],
            'release_duration': [r_dur]
        }

        # randomizing the waveforms
        waves_list = []

        o_list = [oscillators.SineOscillator, oscillators.SquareOscillator, 
            oscillators.TriangleOscillator, oscillators.SawtoothOscillator]
        
        random.shuffle(o_list)

        random_range = np.random.randint(1,5)

        # adding the waveforms to the waves_list for the WaveAdder
        for i in range(random_range):
            freq_detune = np.random.uniform(-0.02, 0.02) # slightly detuning oscillators

            o = o_list.pop() # take one oscillator from o_list

            freq = note_to_hz(note+octave) + note_to_hz(note+octave)*freq_detune

            if (o == oscillators.SineOscillator):
                osc_info['sine_wave'] = [1]
                osc_info['sine_freq'] = [freq]
            elif (o == oscillators.SquareOscillator):
                osc_info['square_wave'] = [1]
                osc_info['square_freq'] = [freq]
            elif (o == oscillators.TriangleOscillator):
                osc_info['triangle_wave'] = [1]
                osc_info['triangle_freq'] = [freq]
            elif (o == oscillators.SawtoothOscillator):
                osc_info['saw_wave'] = [1]
                osc_info['saw_freq'] = [freq]

            waves_list.append(oscillators.ModulatedOscillator(
                o(
                    freq=note_to_hz(note+octave)+note_to_hz(note+octave)*freq_detune, amp=1/random_range),
                    envelopes.ADSREnvelope(attack_duration=a_dur, decay_duration=d_dur, sustain_level=s_lvl,
                        sustain_duration=s_dur, release_duration=r_dur),
                    amp_mod=amp_mod
            )
        )

        osc = WaveAdder(*waves_list) # creating single sound with multiple oscillators 
        
        wav = get_val(iter(osc), 44100 * dur)

        file_name = 'multi_' + note + octave + '_' + str(idx)

        osc_info['file_name'] = file_name

        df = pd.concat([df, pd.DataFrame(osc_info)], ignore_index=True, axis=0) # add to dataframe

        # export .wav file
        wave_to_file(wav, fname=f"audio_files/mono/{folder_name}/wavs/{file_name}.wav")

    # export dataframe as .csv
    df.to_csv(f'audio_files/mono/{folder_name}/file_data{starting_range_index}-{starting_range_index + 1499}.csv')
else: # testing with single samples
    osc = WaveAdder(
        *[oscillators.ModulatedOscillator(
            oscillators.SineOscillator(freq=note_to_hz('A4'), amp=1),
            envelopes.ADSREnvelope(attack_duration=1, decay_duration=1, sustain_level=.5,
                sustain_duration=2, release_duration=2),
            amp_mod=amp_mod, # needed for adsr
            # freq_mod=lambda env, init_freq: freq_mod(env, init_freq, mod_amt=1, sustain_level=0.2)
        ), 
        oscillators.ModulatedOscillator(
            oscillators.SquareOscillator(freq=note_to_hz('A5'), amp=1),
            envelopes.ADSREnvelope(attack_duration=1, decay_duration=1, sustain_level=.5,
                sustain_duration=2, release_duration=2),
            amp_mod=amp_mod, # needed for adsr
            # freq_mod=lambda env, init_freq: freq_mod(env, init_freq, mod_amt=1, sustain_level=0.2)
        )]
    )

    print(oscillators.ModulatedOscillator(
        oscillators.SineOscillator(note_to_hz("A2")),
        ADSREnvelope(0.01, 0.1, 0.4),
        amp_mod=amp_mod
    ),
    oscillators.ModulatedOscillator(
        oscillators.SineOscillator(note_to_hz("A2") + 3),
        ADSREnvelope(0.01, 0.1, 0.4),
        amp_mod=amp_mod
    ),
)

    wav = get_val(iter(osc), 44100 * dur)
    # # wav = to_16(np.array(wav), 0.1)
    wave_to_file(wav, fname="audio_files/mono/one_offs/test1.wav")