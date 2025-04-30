
import streamlit as st
import librosa
import numpy as np
import tempfile
import sounddevice as sd
import scipy.io.wavfile as wav
import os

st.set_page_config(page_title="Absolute Ear", layout="centered")
st.title("🎵 Absolute Ear – Audio to Sheet")

st.markdown("""
Ce site détecte les **notes musicales** à partir d'un fichier audio ou de ta **voix en direct** 🎙️. 
Il affiche ensuite la transcription sous forme de notes simples.
""")

# --- Upload file or record ---
upload_mode = st.radio("Choisis ta source :", ("Uploader un fichier", "Enregistrer depuis le micro"))

AUDIO_DURATION = 5  # durée d'enregistrement micro (secondes)
y = None

if upload_mode == "Uploader un fichier":
    uploaded_file = st.file_uploader("Upload un fichier WAV", type=["wav"])
    if uploaded_file:
        y, sr = librosa.load(uploaded_file, sr=None)
        st.audio(uploaded_file)

elif upload_mode == "Enregistrer depuis le micro":
    if st.button("🎙️ Démarrer l'enregistrement audio (5 sec)"):
        st.info("Enregistrement en cours...")
        fs = 22050
        audio_data = sd.rec(int(AUDIO_DURATION * fs), samplerate=fs, channels=1)
        sd.wait()

        # Sauvegarde dans un fichier temporaire
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
            wav.write(f.name, fs, (audio_data * 32767).astype(np.int16))
            y, sr = librosa.load(f.name, sr=None)
            st.success("Enregistrement terminé !")
            st.audio(f.name)

# --- Analyse et détection des notes ---
if y is not None:
    pitches, mags = librosa.piptrack(y=y, sr=sr)
    notes = []
    for t in range(pitches.shape[1]):
        index = mags[:, t].argmax()
        pitch = pitches[index, t]
        if pitch > 0:
            note = librosa.hz_to_note(pitch)
            notes.append(note)

    st.subheader("🎼 Notes détectées :")
    st.code(" ".join(notes[:100]))  # Affiche les 100 premières notes

    # Audio feedback (jouer une note synthétique de feedback ?)
    if st.button("🔁 Réécouter synthèse des notes détectées"):
        freqs = [librosa.note_to_hz(n) for n in notes[:20]]  # limiter à 20 notes
        synth = np.concatenate([np.sin(2 * np.pi * f * np.linspace(0, 0.3, int(sr * 0.3))) for f in freqs])
        sd.play(synth, samplerate=sr)
        sd.wait()
        st.success("Synthèse jouée avec succès !")
