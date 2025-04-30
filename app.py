import streamlit as st
import librosa
import numpy as np

st.set_page_config(page_title="Absolute Ear", layout="centered")
st.title("🎵 Absolute Ear – Audio to Notes")

st.markdown("""
Ce site détecte les **notes musicales** à partir d’un fichier audio `.wav`.  
📌 Version en ligne : l’enregistrement micro et la lecture audio sont désactivés.
""")

uploaded_file = st.file_uploader("Upload a WAV file", type=["wav"])
if uploaded_file is not None:
    st.audio(uploaded_file)
    y, sr = librosa.load(uploaded_file, sr=None)
    pitches, mags = librosa.piptrack(y=y, sr=sr)

    notes = []
    for t in range(pitches.shape[1]):
        index = mags[:, t].argmax()
        pitch = pitches[index, t]
        if pitch > 0:
            note = librosa.hz_to_note(pitch)
            notes.append(note)

    st.subheader("🎼 Notes détectées :")
    st.code(" ".join(notes[:100]))  # afficher les 100 premières notes

