import librosa
import numpy as np

def Mel_Frequency_Cepstral_Coefficients(audio, sr, n_mfcc=13):
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
    return np.mean(mfccs, axis=1)

def Centroid(audio, sr, frame_length=None, hop_length=None):
    if frame_length is None:
        frame_length = int(0.95 * sr)  
    if hop_length is None:
        hop_length = int(0.6 * frame_length) 
    spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=sr, n_fft=frame_length, hop_length=hop_length)
    return np.mean(spectral_centroid)

def Spread(audio, sr, frame_length=None, hop_length=None):
    if frame_length is None:
        frame_length = int(0.95 * sr)
    if hop_length is None:
        hop_length = int(0.6 * frame_length)
    spectral_spread = librosa.feature.spectral_bandwidth(
        y=audio, 
        sr=sr, 
        n_fft=frame_length, 
        hop_length=hop_length
    )
    return np.mean(spectral_spread)

def Skewness(audio, sr, frame_length=None, hop_length=None):
    if frame_length is None:
        frame_length = int(0.95 * sr)
    if hop_length is None:
        hop_length = int(0.6 * frame_length)

    eps = 1e-10
    
    spec = np.abs(librosa.stft(audio, n_fft=frame_length, hop_length=hop_length))
    frequencies = librosa.fft_frequencies(sr=sr, n_fft=frame_length)
    
    spec_norm = spec / (np.sum(spec, axis=0) + eps)
    
    centroid = np.sum(frequencies.reshape(-1, 1) * spec_norm, axis=0)
    spread = np.sqrt(np.sum(np.power(frequencies.reshape(-1, 1) - centroid, 2) * spec_norm, axis=0) + eps)
    skewness = np.sum(np.power(frequencies.reshape(-1, 1) - centroid, 3) * spec_norm, axis=0) / (np.power(spread, 3) + eps)
    
    return np.nan_to_num(np.mean(skewness), nan=0.0, posinf=0.0, neginf=0.0)

def Kurtosis(audio, sr, frame_length=None, hop_length=None):
    if frame_length is None:
        frame_length = int(0.95 * sr)
    if hop_length is None:
        hop_length = int(0.6 * frame_length)

    eps = 1e-10
    spec = np.abs(librosa.stft(audio, n_fft=frame_length, hop_length=hop_length))
    frequencies = librosa.fft_frequencies(sr=sr, n_fft=frame_length)
    
    spec_norm = spec / (np.sum(spec, axis=0) + eps)
    
    centroid = np.sum(frequencies.reshape(-1, 1) * spec_norm, axis=0)
    spread = np.sqrt(np.sum(np.power(frequencies.reshape(-1, 1) - centroid, 2) * spec_norm, axis=0) + eps)
    kurtosis = (np.sum(np.power(frequencies.reshape(-1, 1) - centroid, 4) * spec_norm, axis=0) / (np.power(spread, 4) + eps)) - 3
    
    return np.nan_to_num(np.mean(kurtosis), nan=0.0, posinf=0.0, neginf=0.0)

def Slope(audio, sr, frame_length=None, hop_length=None):
    if frame_length is None:
        frame_length = int(0.95 * sr)
    if hop_length is None:
        hop_length = int(0.6 * frame_length)

    spec = np.abs(librosa.stft(audio, n_fft=frame_length, hop_length=hop_length))
    frequencies = librosa.fft_frequencies(sr=sr, n_fft=frame_length)
    
    freq_norm = (frequencies - np.min(frequencies)) / (np.max(frequencies) - np.min(frequencies))
    
    slopes = []
    for frame in range(spec.shape[1]):
        spec_frame = spec[:, frame]
        spec_norm = spec_frame / (np.max(spec_frame) + 1e-10)
        
        try:
            coeffs = np.polyfit(freq_norm, spec_norm, deg=1)
            slopes.append(coeffs[0])
        except:
            slopes.append(0.0)
    
    return np.nan_to_num(np.mean(slopes), nan=0.0, posinf=0.0, neginf=0.0)

def Decrease(audio, sr, frame_length=None, hop_length=None):
    if frame_length is None:
        frame_length = int(0.95 * sr)
    if hop_length is None:
        hop_length = int(0.6 * frame_length)

    spec = np.abs(librosa.stft(audio, n_fft=frame_length, hop_length=hop_length))
    eps = 1e-10
    
    decreases = []
    for frame in range(spec.shape[1]):
        spectrum = spec[:, frame]
        spectrum = spectrum / (np.max(spectrum) + eps)
        
        if len(spectrum) > 1:  
            k = np.arange(1, len(spectrum))
            try:
                decrease = np.sum((spectrum[1:] - spectrum[0]) / k) / (np.sum(spectrum[1:]) + eps)
                decreases.append(decrease)
            except:
                decreases.append(0.0)
        else:
            decreases.append(0.0)
    
    return np.nan_to_num(np.mean(decreases), nan=0.0, posinf=0.0, neginf=0.0)

def Roll_Off(audio, sr, roll_percent=0.85):
    rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr, roll_percent=roll_percent)
    return np.mean(rolloff) 

def Flux_mean(audio, sr):
    spectral_flux = librosa.onset.onset_strength(y=audio, sr=sr)
    return np.mean(spectral_flux)

def Flux_variance(audio, sr):
    spectral_flux = librosa.onset.onset_strength(y=audio, sr=sr)
    return np.var(spectral_flux)
    
def MPEG7_Centroid(audio, sr):
    eps = 1e-10
    spectrum = np.abs(np.fft.rfft(audio))
    frequencies = np.fft.rfftfreq(len(audio), d=1/sr)
    return np.nan_to_num(np.sum(frequencies * spectrum) / (np.sum(spectrum) + eps), nan=0.0, posinf=0.0, neginf=0.0)

def MPEG7_Spread(audio, sr, centroid):
    eps = 1e-10
    spectrum = np.abs(np.fft.rfft(audio))
    frequencies = np.fft.rfftfreq(len(audio), d=1/sr)
    numerator = np.sum(((frequencies - centroid)**2) * spectrum)
    denominator = np.sum(spectrum) + eps
    return np.nan_to_num(np.sqrt(numerator / denominator), nan=0.0, posinf=0.0, neginf=0.0)

def MPEG7_Flatness(audio):
    eps = 1e-10
    spectrum = np.abs(np.fft.rfft(audio)) + eps
    geometric_mean = np.exp(np.mean(np.log(spectrum)))
    arithmetic_mean = np.mean(spectrum)
    return np.nan_to_num(geometric_mean / arithmetic_mean, nan=0.0, posinf=0.0, neginf=0.0)

def Pitch(audio, sr):
    pitch, voiced_flag, voiced_probs = librosa.pyin(
        audio,
        fmin=librosa.note_to_hz('C1'),  
        fmax=librosa.note_to_hz('B7'),  
        sr=sr
    )

    valid_pitch = pitch[~np.isnan(pitch)]
    
    if len(valid_pitch) > 0:
        return np.mean(valid_pitch)
    else:
        return 0.0
    
def Zero_Crossing_Rate(audio, sr, frame_length=None, hop_length=None):
    if frame_length is None:
        frame_length = int(0.95 * sr)  
    if hop_length is None:
        hop_length = int(0.95 * frame_length) 
        
    zcr = librosa.feature.zero_crossing_rate(
        y=audio,
        frame_length=frame_length,
        hop_length=hop_length
    )
    return np.mean(zcr)

def Log_Attack_Time(audio, sr, attack_threshold=0.5):
    try:
        envelope = np.abs(audio)
        envelope = librosa.util.normalize(envelope)
        attack_time_index = np.where(envelope >= attack_threshold)[0][0]
        attack_time = attack_time_index / sr
        return np.nan_to_num(np.log10(attack_time + 1e-10), nan=0.0, posinf=0.0, neginf=0.0)
    except (IndexError, ValueError):
        return 0.0 

def AM_Index(audio):
    eps = 1e-10
    envelope = np.abs(audio)
    a_max = np.max(envelope)
    a_min = np.min(envelope)
    if a_max + a_min < eps:
        return 0
    return np.nan_to_num((a_max - a_min) / (a_max + a_min), nan=0.0, posinf=0.0, neginf=0.0)

#descriptors = ["Mel-Frequency Cepstal Coefficients", "Centroid", "Spread", "Skewness", "Kurtosis", "Slope", "Decrease", "Roll-off", "Flux (mean)", "Flux (variance)", "MPEG7_Centroid", "MPEG7_Spread", "MPEG7_Flatness", "Pitch", "Zero-Crossing Rate", "Log-Attack Time", "AM Index"]
def calculate_feature(audio, sr, descriptor):
    if descriptor == "Mel-Frequency Cepstal Coefficients":
        return Mel_Frequency_Cepstral_Coefficients(audio, sr)
    elif descriptor == "Centroid":  
        return Centroid(audio, sr)
    elif descriptor == "Spread": 
        return Spread(audio, sr)
    elif descriptor == "Skewness":
        return Skewness(audio, sr)
    elif descriptor == "Kurtosis":
        return Kurtosis(audio, sr)
    elif descriptor == "Slope":
        return Slope(audio, sr)
    elif descriptor == "Decrease":
        return Decrease(audio, sr)
    elif descriptor == "Roll-off":  
        return Roll_Off(audio, sr)
    elif descriptor == "Flux (mean)":  
        return Flux_mean(audio, sr)
    elif descriptor == "Flux (variance)":  
        return Flux_variance(audio, sr)
    elif descriptor == "MPEG7_Centroid":
        return MPEG7_Centroid(audio, sr)
    elif descriptor == "MPEG7_Spread":
        centroid = MPEG7_Centroid(audio, sr)
        return MPEG7_Spread(audio, sr, centroid)
    elif descriptor == "MPEG7_Flatness":
        return MPEG7_Flatness(audio)
    elif descriptor == "Pitch":  
        return Pitch(audio, sr)
    elif descriptor == "Zero-Crossing Rate": 
        return Zero_Crossing_Rate(audio, sr)
    elif descriptor == "Log-Attack Time":
        return Log_Attack_Time(audio, sr)
    elif descriptor == "AM Index":
        return AM_Index(audio)
    else:
        raise ValueError("Invalid descriptor")
