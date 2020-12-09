import os
import re
import librosa
import numpy as np
import sklearn
from entropy import spectral_entropy

from library.mapper import mapper_chest_loc, mapper_sex, mapper_acquisition, mapper_diagnosis, mapper_rec_equipment


def preprocess_signal(x):
    # Scale [0,1]
    x = sklearn.preprocessing.minmax_scale(x, axis=0)
    # Pre-emphasis
    x = librosa.effects.preemphasis(x)
    return x

def chunks(l, k):
  '''
  Yields chunks of size k from a given list.
  '''
  for i in range(0, len(l), k):
    yield l[i:i+k]


def extract_audio_features(data, freq, stft):
    '''
    Describes spectral envelope, in music information retrieval: timbre = zven (slo)
    Generally, we use first 13 coefficients. The envelope of the time power spectrum
    of the speech signal is representative of the vocal tract
    '''
    mfcc = librosa.feature.mfcc(y=data, sr=freq, n_mfcc=13)

    '''
    Measure of number of times in a given time interval/frame that the amplitude of the speech
    signals passes through a  value of zero. Classifying percussive sounds - ours is not?
    '''
    zero_crossing_rate = librosa.feature.zero_crossing_rate(y=data)

    '''
    The energy of a signal corresponds to the total magntiude of the signal. For audio signals, 
    that roughly corresponds to how loud the signal is. We calculate energy using RMS.
    '''
    rms = librosa.feature.rms(y=data)

    '''
    A chroma vector is a typically a 12-element feature vector indicating how much energy of 
    each pitch class, {C, C#, D, D#, E, ..., B}, is present in the signal.
    '''
    chroma = librosa.feature.chroma_stft(S=stft, sr=freq, n_fft=512)

    '''
    Chroma energy normalized statistics (CENS). The main idea of CENS features is that taking
    statistics over large windows smooths local deviations in tempo, articulation, and musical ornaments 
    such as trills and arpeggiated chords. CENS are best used for tasks such as audio matching and similarity.
    '''
    cens = librosa.feature.chroma_cens(y=data, sr=freq)

    '''
    The spectral centroid indicates at which frequency the energy of a spectrum is centered upon. 
    This is like a weighted mean.
    '''
    spectral_centroid = librosa.feature.spectral_centroid(y=data, sr=freq, n_fft=512)

    '''
    Difference between highest and lowest frequency in a frame.
    '''
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=data, sr=freq, n_fft=512)

    '''
    Spectral contrast considers the spectral peak, the spectral valley, and their difference in each frequency subband.
    '''
    spectral_contrast = librosa.feature.spectral_contrast(y=data, sr=freq, n_fft=512)

    '''
    Spectral flatness is the measure of uniformity in the frequency distribution of the power spectrum, calculated as 
    the ratio between geometric and arithmetic mean. and can be used to distinguish between noise and harmonic sounds, 
    which can be very important in respiratory sound analysis, since breathing tends to be less harmonic sound, if 
    wheezes are not present.
    '''
    spectral_flatness = librosa.feature.spectral_flatness(y=data, n_fft=512)

    '''
    Spectral rolloff is the frequency below which a specified percentage of the total spectral energy, e.g. 85%, lies.
    '''
    spectral_rolloff = librosa.feature.spectral_rolloff(y=data, sr=freq, n_fft=512)

    '''
    Tonnetz computes the tonal centroid features and can be used to show harmonic relationships in an audio signal. Tonal 
    features are often used for harmonic sounds classification.
    '''
    tonnetz = librosa.feature.tonnetz(y=librosa.effects.harmonic(data), sr=freq)

    '''
    Entropy of energy - we first divide windows into sub-frames and evaluate the probability of the energy in a sub-frame occuring.
    '''
    entropy = spectral_entropy(x=data, sf=freq)

    return {
        'MFCC': np.mean(mfcc, axis=1).tolist() + np.std(mfcc, axis=1).tolist(),
        'CHROMA': np.mean(chroma, axis=1).tolist() + np.std(chroma, axis=1).tolist(),
        'CENS': np.mean(cens, axis=1).tolist() + np.std(cens, axis=1).tolist(),
        'SPECTRAL_CENTROID': np.mean(spectral_centroid, axis=1).tolist() + np.std(spectral_centroid, axis=1).tolist(),
        'SPECTRAL_BANDWIDTH': np.mean(spectral_bandwidth, axis=1).tolist() + np.std(spectral_bandwidth, axis=1).tolist(),
        'SPECTRAL_CONTRAST': np.mean(spectral_contrast, axis=1).tolist() + np.std(spectral_contrast, axis=1).tolist(),
        'SPECTRAL_FLATNESS': np.mean(spectral_flatness, axis=1).tolist() + np.std(spectral_flatness, axis=1).tolist(),
        'SPECTRAL_ROLLOFF': np.mean(spectral_rolloff, axis=1).tolist() + np.std(spectral_rolloff, axis=1).tolist(),
        'RMS': np.mean(rms, axis=1).tolist() + np.std(rms, axis=1).tolist(),
        'ZERO_CROSSING_RATE': np.mean(zero_crossing_rate, axis=1).tolist() + np.std(zero_crossing_rate, axis=1).tolist(),
        'TONNETZ': np.mean(tonnetz, axis=1).tolist() + np.std(tonnetz, axis=1).tolist(),
        'ENTROPY': [entropy.astype(float)]
    }

def read_data(path):
    patient_data = {}

    file_demo_info = open(f'{path}demographic_info.txt', 'r')
    pattern = re.compile(r'\s*')

    for line in file_demo_info.readlines():
        demo_info = line.split(' ')
        demo_info = list(map(lambda x: re.sub(pattern, '', x), demo_info))

        if len(demo_info) < 6:
            continue
        bmi = None
        if demo_info[4] != 'NA' and demo_info[5] != 'NA':
            weight = float(demo_info[4])
            height = float(demo_info[5])
            bmi = weight / (height / 100) ** 2
        if demo_info[3] != 'NA':
            bmi = float(demo_info[3])
        patient_data[str(demo_info[0])] = {
            'AGE': None if demo_info[1] == 'NA' else float(demo_info[1]),
            'SEX': None if demo_info[2] == 'NA' else mapper_sex(demo_info[2]),
            'BMI': bmi,
            'RECORDINGS': []
        }

    file_demo_info = open(f'{path}patient_diagnosis.csv', 'r')
    for line in file_demo_info.readlines():
        diagnosis_info = line.split(',')
        diagnosis_info = list(map(lambda x: re.sub(pattern, '', x), diagnosis_info))
        patient_data[diagnosis_info[0]]['DIAGNOSIS'] = None if diagnosis_info[1] == 'NA' else mapper_diagnosis(
            diagnosis_info[1])

    path = f'{path}audio_and_txt_files'
    for audio_file in os.listdir(path):
        f_name_type = audio_file.split('.')
        pk_filename = f_name_type[0]
        file_type = f_name_type[1]

        if file_type == 'txt':
            patient, recording = read_rec_txt(path, pk_filename)
            patient_data[patient]['RECORDINGS'].append(recording)

    # For each patient
    for patient_id in patient_data.keys():
        print(patient_id)
        # For each recording
        for ix in range(len(patient_data[patient_id]['RECORDINGS'])):
            # For each respiratory cycle
            recording = patient_data[patient_id]['RECORDINGS'][ix]
            for cycle in recording['RESPIRATORY_CYCLES']:
                data, freq = librosa.load(f'{path}/{recording["REC_IX"]}.wav', offset=cycle['START'],
                                                   duration=cycle['END'] - cycle['START'])

                data = preprocess_signal(data)

                stft = np.abs(librosa.stft(data, n_fft=512))

                # Extract features from data wave and spectrogram
                cycle['DATA'] = extract_audio_features(data, freq, stft)
    return patient_data


def read_rec_txt(path, filename):
    properties = filename.split('_')
    pattern = re.compile(r'\s*')

    file_resp_cycles = open(f'{path}/{filename}.txt', 'r')

    respiratory_cycles = []
    for line in file_resp_cycles.readlines():
        cycle = line.split('\t')
        cycle = list(map(lambda x: re.sub(pattern, '', x), cycle))

        respiratory_cycles.append({
            'START': float(cycle[0]),
            'END': float(cycle[1]),
            'CRACKLES': cycle[2],
            'WHEEZES': cycle[3]
        })
    return properties[0], {
        'REC_IX': filename,
        'CHEST_LOC': mapper_chest_loc(properties[2]),
        'ACQUISITION': mapper_acquisition(properties[3]),
        'REC_EQUIPMENT': mapper_rec_equipment(properties[4]),
        'RESPIRATORY_CYCLES': respiratory_cycles
    }