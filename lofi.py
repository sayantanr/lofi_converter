import streamlit as st
import librosa
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter

st.set_page_config(layout="wide")
st.title("🎧 Ultimate Bangla / Hindi Lo-Fi Studio")

# ===============================
# File Upload
# ===============================
file = st.file_uploader("Upload MP3 song", type=["mp3"])

# ===============================
# CONTROLS
# ===============================
st.sidebar.header("🎛 Controls")

tempo = st.sidebar.slider("Tempo", 0.5, 1.2, 0.85)
pitch = st.sidebar.slider("Pitch (semitones)", -6, 6, -1)
low_cut = st.sidebar.slider("Low-pass cutoff (Hz)", 800, 8000, 3000)
high_cut = st.sidebar.slider("High-pass cutoff (Hz)", 20, 500, 80)

noise_amt = st.sidebar.slider("Vinyl Noise", 0.0, 0.02, 0.003)
hiss_amt = st.sidebar.slider("Tape Hiss", 0.0, 0.01, 0.002)

reverb_amt = st.sidebar.slider("Reverb", 0.0, 1.0, 0.2)
delay_amt = st.sidebar.slider("Delay", 0.0, 0.5, 0.1)

stereo_width = st.sidebar.slider("Stereo Width", 0.5, 2.0, 1.2)
pan = st.sidebar.slider("Pan (-L to +R)", -1.0, 1.0, 0.0)

fade_time = st.sidebar.slider("Fade In/Out (sec)", 0.0, 5.0, 1.0)

normalize = st.sidebar.checkbox("Normalize", True)
show_wave = st.sidebar.checkbox("Show Waveform", True)
show_spec = st.sidebar.checkbox("Show Spectrogram", False)

out_format = st.sidebar.selectbox("Output format", ["mp3", "wav"])

# ===============================
# DSP HELPERS
# ===============================
def lowpass(y, sr, cutoff):
    b, a = butter(6, cutoff / (sr / 2), btype="low")
    return lfilter(b, a, y)

def highpass(y, sr, cutoff):
    b, a = butter(4, cutoff / (sr / 2), btype="high")
    return lfilter(b, a, y)

def fade(y, sr, t):
    n = int(sr * t)
    if n == 0:
        return y
    fade_in = np.linspace(0, 1, n)
    fade_out = np.linspace(1, 0, n)
    y[:n] *= fade_in
    y[-n:] *= fade_out
    return y

# ===============================
# PROCESS
# ===============================
if file:
    y, sr = librosa.load(file, mono=False)
    original = y.copy()

    # Mono safety
    if y.ndim > 1:
        y = librosa.to_mono(y)

    # Time + Pitch (SAFE API)
    y = librosa.effects.time_stretch(y, rate=tempo)
    y = librosa.effects.pitch_shift(y, sr=sr, n_steps=pitch)

    # Filters
    y = lowpass(y, sr, low_cut)
    y = highpass(y, sr, high_cut)

    # Texture
    y += noise_amt * np.random.randn(len(y))
    y += hiss_amt * np.random.randn(len(y))

    # Reverb & delay (simple)
    y = y * (1 - reverb_amt) + np.roll(y, int(sr * 0.03)) * reverb_amt
    y = y + delay_amt * np.roll(y, int(sr * 0.25))

    # Stereo widening
    left = y * (1 - pan)
    right = y * (1 + pan)
    stereo = np.vstack([left, right]) * stereo_width

    # Fade
    stereo[0] = fade(stereo[0], sr, fade_time)
    stereo[1] = fade(stereo[1], sr, fade_time)

    # Normalize
    if normalize:
        stereo /= np.max(np.abs(stereo))

    # Save
    out_file = f"lofi.{out_format}"
    sf.write(out_file, stereo.T, sr)

    # ===============================
    # OUTPUT
    # ===============================
    st.subheader("🎵 Lo-Fi Preview")
    st.audio(out_file)

    st.download_button("⬇ Download", open(out_file, "rb"), out_file)

    # ===============================
    # VISUALS
    # ===============================
    if show_wave:
        st.subheader("📈 Waveform (Lo-Fi)")
        st.line_chart(stereo[0][:6000])

    if show_spec:
        st.subheader("🌈 Spectrogram")
        fig, ax = plt.subplots()
        D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
        img = librosa.display.specshow(D, sr=sr, x_axis="time", y_axis="hz", ax=ax)
        fig.colorbar(img, ax=ax)
        st.pyplot(fig)
