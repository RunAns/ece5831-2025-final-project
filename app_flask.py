from flask import Flask, render_template, request
import os
import math
import tempfile
import subprocess

import numpy as np
import librosa
import keras  # or from tensorflow import keras

# -----------------------
# App setup
# -----------------------
app = Flask(__name__)

MODEL_PATH = "MusicGenre_CNN_79.73.h5"
model = keras.models.load_model(MODEL_PATH)

# -----------------------
# Constants (match training)
# -----------------------
SAMPLE_RATE = 22050
NUM_MFCC = 13
N_FFT = 2048
HOP_LENGTH = 512
TRACK_DURATION = 30
NUM_SEGMENTS = 10

SAMPLES_PER_TRACK = SAMPLE_RATE * TRACK_DURATION
SAMPLES_PER_SEGMENT = int(SAMPLES_PER_TRACK / NUM_SEGMENTS)
EXPECTED_FRAMES = int(math.ceil(SAMPLES_PER_SEGMENT / HOP_LENGTH))

genre_dict = {
    0: "disco",
    1: "pop",
    2: "classical",
    3: "metal",
    4: "rock",
    5: "blues",
    6: "hiphop",
    7: "reggae",
    8: "country",
    9: "jazz",
}


def convert_to_wav_if_needed(input_path: str) -> str:
    """
    Converts non-wav audio to wav using ffmpeg.
    Returns wav path. If already wav, returns input_path.
    """
    ext = os.path.splitext(input_path)[1].lower()
    if ext == ".wav":
        return input_path

    wav_path = os.path.splitext(input_path)[0] + "_converted.wav"
    cmd = ["ffmpeg", "-y", "-i", input_path, "-ac", "1", "-ar", str(SAMPLE_RATE), wav_path]
    subprocess.run(cmd, check=True)
    return wav_path


def extract_mfcc_segments(audio_path: str) -> np.ndarray:
    """
    Output: (segments, 130, 13, 1) float32
    """
    y, sr = librosa.load(audio_path, sr=SAMPLE_RATE, mono=True)

    segments = []
    for d in range(NUM_SEGMENTS):
        start = d * SAMPLES_PER_SEGMENT
        finish = start + SAMPLES_PER_SEGMENT
        if finish > len(y):
            continue

        mfcc = librosa.feature.mfcc(
            y=y[start:finish],
            sr=sr,
            n_mfcc=NUM_MFCC,
            n_fft=N_FFT,
            hop_length=HOP_LENGTH
        ).T  # (frames, 13)

        # pad/crop to EXPECTED_FRAMES to match training shape (usually 130)
        frames = mfcc.shape[0]
        if frames < EXPECTED_FRAMES:
            pad = np.zeros((EXPECTED_FRAMES - frames, NUM_MFCC), dtype=mfcc.dtype)
            mfcc = np.vstack([mfcc, pad])
        else:
            mfcc = mfcc[:EXPECTED_FRAMES, :]

        segments.append(mfcc)

    if len(segments) == 0:
        raise ValueError("No valid segments extracted from the audio.")

    X = np.stack(segments, axis=0).astype(np.float32)  # (segments, frames, mfcc)
    X = X[..., np.newaxis]  # (segments, frames, mfcc, 1)
    return X


def predict_top3(X_segments: np.ndarray):
    """
    X_segments: (segments, 130, 13, 1)
    Returns: (pred_label, pred_prob, second_label, second_prob, third_label, third_prob)
    """
    probs = model.predict(X_segments, verbose=0)      # (segments, 10)
    avg_probs = probs.mean(axis=0)                   # (10,)

    order = avg_probs.argsort()                      # ascending
    top1, top2, top3 = order[-1], order[-2], order[-3]

    return (
        genre_dict[int(top1)], float(avg_probs[top1]),
        genre_dict[int(top2)], float(avg_probs[top2]),
        genre_dict[int(top3)], float(avg_probs[top3]),
    )


@app.route("/")
def homepage():
    title = "MGC"
    return render_template("homepage.html", title=title)


@app.route("/prediction", methods=["POST"])
def prediction():
    title = "MGC | Prediction"

    # Template form should send a file input named "myfile"
    if "myfile" not in request.files:
        return render_template("prediction.html", title=title, error="No file uploaded.")

    f = request.files["myfile"]
    if f.filename.strip() == "":
        return render_template("prediction.html", title=title, error="Empty filename.")

    suffix = os.path.splitext(f.filename)[1].lower()

    with tempfile.TemporaryDirectory() as tmpdir:
        input_path = os.path.join(tmpdir, "upload" + suffix)
        f.save(input_path)

        try:
            wav_path = convert_to_wav_if_needed(input_path)
            X = extract_mfcc_segments(wav_path)
            (p1, s1, p2, s2, p3, s3) = predict_top3(X)

            return render_template(
                "prediction.html",
                title=title,
                prediction=p1,
                probability=f"{s1*100:.2f}",
                second_prediction=p2,
                second_probability=f"{s2*100:.2f}",
                third_prediction=p3,
                third_probability=f"{s3*100:.2f}",
                segments_used=X.shape[0],
                input_shape=str(X.shape),
            )

        except subprocess.CalledProcessError:
            return render_template(
                "prediction.html",
                title=title,
                error="ffmpeg failed. Make sure ffmpeg is installed and available in PATH."
            )
        except Exception as e:
            return render_template(
                "prediction.html",
                title=title,
                error=f"Prediction failed: {type(e).__name__}: {e}"
            )


@app.route("/about")
def about():
    title = "MGC | About"
    return render_template("about.html", title=title)


@app.route("/project")
def project():
    title = "MGC | Project"
    return render_template("project.html", title=title)


if __name__ == "__main__":
    app.run(debug=True)
