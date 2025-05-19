import streamlit as st
import parselmouth
import joblib
from parselmouth.praat import call
import numpy as np
import scipy.signal
import scipy.stats
import tempfile

lda_model = joblib.load('/Users/tanisha/Desktop/integrate/new_parkinsons_model.pkl')

def extract_features(audio_path):
    try:
        # Load the audio file
        sound = parselmouth.Sound(audio_path)

        # Extract fundamental frequency (pitch)
        pitch = call(sound, "To Pitch", 0.0, 75, 500)
        mean_F0 = float(call(pitch, "Get mean", 0, 0, "Hertz") or 0)
        min_F0 = float(call(pitch, "Get minimum", 0, 0, "Hertz", "Parabolic") or 0)
        max_F0 = float(call(pitch, "Get maximum", 0, 0, "Hertz", "Parabolic") or 0)

        # Jitter Features
        pointProcess = call(sound, "To PointProcess (periodic, cc)", 75, 500)
        localJitter = float(call(pointProcess, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3) or 0)
        localabsoluteJitter = float(call(pointProcess, "Get jitter (local, absolute)", 0, 0, 0.0001, 0.02, 1.3) or 0)
        rapJitter = float(call(pointProcess, "Get jitter (rap)", 0, 0, 0.0001, 0.02, 1.3) or 0)
        ppqJitter = float(call(pointProcess, "Get jitter (ppq5)", 0, 0, 0.0001, 0.02, 1.3) or 0)
        ddpJitter = float(3 * rapJitter)

        # Shimmer Features
        localShimmer = float(call([sound, pointProcess], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6) or 0)
        localdbShimmer = (float(call([sound, pointProcess], "Get shimmer (local_dB)", 0, 0, 0.0001, 0.02, 1.3, 1.6) or 0))/10
        apq3Shimmer = (float(call([sound, pointProcess], "Get shimmer (apq3)", 0, 0, 0.0001, 0.02, 1.3, 1.6) or 0))/10
        apq5Shimmer = (float(call([sound, pointProcess], "Get shimmer (apq5)", 0, 0, 0.0001, 0.02, 1.3, 1.6) or 0))/10
        apq11Shimmer = (float(call([sound, pointProcess], "Get shimmer (apq11)", 0, 0, 0.0001, 0.02, 1.3, 1.6) or 0))/10
        ddaShimmer = (float(3 * apq3Shimmer))/10

        # Harmonics-to-Noise Ratio (HNR) and Noise-to-Harmonics Ratio (NHR)
        harmonicity = call(sound, "To Harmonicity (cc)", 0.01, 75, 0.1, 1.0)
        hnr = float(call(harmonicity, "Get mean", 0, 0) or 0)
        nhr = 1 / hnr if hnr > 0 else 0  # Avoid division by zero

        # Recurrence Period Density Entropy (RPDE)
        def calculate_rpde(pitch_values):
            if len(pitch_values) < 2:
                return 0
            pitch_diff = np.diff(pitch_values)
            entropy = scipy.stats.entropy(np.histogram(pitch_diff, bins=10)[0])
            return entropy if not np.isnan(entropy) else 0

        pitch_values = pitch.selected_array['frequency']
        rpde = float(calculate_rpde(pitch_values) if np.any(pitch_values) else 0)
        rpde = rpde * 10

        # Detrended Fluctuation Analysis (DFA)
        def calculate_dfa(signal):
            if len(signal) < 4:
                return 0  # Avoid short sequences
            return np.polyfit(range(len(signal)), signal, 1)[0]

        dfa = float(calculate_dfa(pitch_values) if np.any(pitch_values) else 0)
        dfa = dfa * -10

        # Spectral Features: Spread1, Spread2, D2, PPE
        def spectral_features(signal, fs):
            if len(signal) < 2:
                return 0, 0, 0, 0
            freqs, psd = scipy.signal.welch(signal, fs=fs, nperseg=min(1024, len(signal)))
            psd_norm = psd / np.sum(psd)
            mean_freq = np.sum(freqs * psd_norm)
            spread1 = -((np.sqrt(np.sum(((freqs - mean_freq) ** 2) * psd_norm))) / 100)
            spread2 = (np.sqrt(np.sum(((freqs - np.median(freqs)) ** 2) * psd_norm))) / 1000
            ppe = np.sum(psd_norm ** 2)
            d2 = -np.sum(psd_norm * np.log(psd_norm + 1e-10))  # Prevent log(0)
            return float(spread1), float(spread2), float(d2), float(ppe)

        spread1, spread2, d2, ppe = spectral_features(pitch_values, sound.sampling_frequency) if np.any(pitch_values) else (0, 0, 0, 0)

        # Assemble the features into a dictionary
        features = {
            'MDVP:Fo(Hz)': mean_F0,
            'MDVP:Fhi(Hz)': max_F0,
            'MDVP:Flo(Hz)': min_F0,
            'MDVP:Jitter(%)': localJitter,
            'MDVP:Jitter(Abs)': localabsoluteJitter,
            'MDVP:RAP': rapJitter,
            'MDVP:PPQ': ppqJitter,
            'Jitter:DDP': ddpJitter,
            'MDVP:Shimmer': localShimmer,
            'MDVP:Shimmer(dB)': localdbShimmer,
            'Shimmer:APQ3': apq3Shimmer,
            'Shimmer:APQ5': apq5Shimmer,
            'MDVP:APQ': apq11Shimmer,
            'Shimmer:DDA': ddaShimmer,
            'NHR': nhr,
            'HNR': hnr,
            'RPDE': rpde,
            'DFA': dfa,
            'spread1': spread1,
            'spread2': spread2,
            'D2': d2,
            'PPE': ppe,
        }

        return {k: (v if not np.isnan(v) else 0) for k, v in features.items()}  # Replace NaN with 0

    except Exception as e:
        st.error(f"Error processing audio: {e}")
        return {}

st.title("Parkinson's Detection Using Audio Analysis")

audio_file = st.file_uploader("Upload an audio file", type=["wav", "mp3"])
if audio_file is not None:
    # Save the file temporarily to a path
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
        temp_audio.write(audio_file.read())
        temp_audio_path = temp_audio.name

    features = extract_features(temp_audio_path)
    if features:
        values_list = [value for key, value in features.items()]
        array = np.array(values_list)
        reshaped_array = array.reshape(1, 22)

        try:
            prediction = lda_model.predict(reshaped_array)
            if prediction[0] == 1:
                st.write("🩺 Parkinson's Detected")
            else:
                st.write("✅ Healthy")
        except Exception as e:
            st.error(f"⚠️ Error processing file: {e}")
