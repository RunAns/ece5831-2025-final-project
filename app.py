from flask import Flask, render_template, request
import os
import math
import tempfile
import subprocess

import numpy as np
import librosa
import keras  # or: from tensorflow import keras

app = Flask(__name__)

# -----------------------
# Model
# -----------------------
MODEL_PATH = "MusicGenre_CNN_.h5"
model = keras.models.load_model(MODEL_PATH)

# -----------------------
# Audio / feature params (must match training)
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
    0: "disco", 1: "pop", 2: "classical", 3: "metal", 4: "rock",
    5: "blues", 6: "hiphop", 7: "reggae", 8: "country", 9: "jazz",
}


def convert_to_wav_if_needed(input_path: str) -> str:
    """Convert non-wav audio to wav using ffmpeg. Returns wav path."""
    ext = os.path.splitext(input_path)[1].lower()
    if ext == ".wav":
        return input_path

    wav_path = os.path.splitext(input_path)[0] + "_converted.wav"
    cmd = ["ffmpeg", "-y", "-i", input_path, "-ac", "1", "-ar", str(SAMPLE_RATE), wav_path]
    subprocess.run(cmd, check=True)
    return wav_path


def load_and_pad_to_30s(audio_path: str) -> np.ndarray:
    """
    Load audio as mono float, resampled. If shorter than 30s, pad with zeros.
    If longer, trim to 30s.
    """
    y, _ = librosa.load(audio_path, sr=SAMPLE_RATE, mono=True)
    if len(y) < SAMPLES_PER_TRACK:
        y = np.pad(y, (0, SAMPLES_PER_TRACK - len(y)), mode="constant")
    else:
        y = y[:SAMPLES_PER_TRACK]
    return y


def extract_mfcc_segments(audio_path: str) -> np.ndarray:
    """
    Returns X shaped (NUM_SEGMENTS, EXPECTED_FRAMES, NUM_MFCC, 1) float32.
    Always returns 10 segments by padding/trimming audio to 30 seconds.
    """
    y = load_and_pad_to_30s(audio_path)

    segments = []
    for d in range(NUM_SEGMENTS):
        start = d * SAMPLES_PER_SEGMENT
        finish = start + SAMPLES_PER_SEGMENT

        mfcc = librosa.feature.mfcc(
            y=y[start:finish],
            sr=SAMPLE_RATE,
            n_mfcc=NUM_MFCC,
            n_fft=N_FFT,
            hop_length=HOP_LENGTH
        ).T  # (frames, 13)

        # enforce fixed frame length
        frames = mfcc.shape[0]
        if frames < EXPECTED_FRAMES:
            pad = np.zeros((EXPECTED_FRAMES - frames, NUM_MFCC), dtype=mfcc.dtype)
            mfcc = np.vstack([mfcc, pad])
        else:
            mfcc = mfcc[:EXPECTED_FRAMES, :]

        segments.append(mfcc)

    X = np.stack(segments, axis=0).astype(np.float32)  # (10, frames, 13)
    X = X[..., np.newaxis]  # (10, frames, 13, 1)
    return X


def predict_top3(X_segments: np.ndarray):
    """
    X_segments: (10, frames, 13, 1)
    Returns (top1_name, top1_prob, top2_name, top2_prob, top3_name, top3_prob)
    using averaged segment probabilities.
    """
    probs = model.predict(X_segments, verbose=0)  # (10, 10)
    avg = probs.mean(axis=0)                     # (10,)

    order = avg.argsort()
    top1, top2, top3 = int(order[-1]), int(order[-2]), int(order[-3])

    return (
        genre_dict[top1], float(avg[top1]),
        genre_dict[top2], float(avg[top2]),
        genre_dict[top3], float(avg[top3]),
    )


# -----------------------
# Routes (clean minimal frontend)
# -----------------------
@app.route("/")
def homepage():
    return render_template("index.html")


@app.route("/prediction", methods=["POST"])
def prediction():
    title = "MGC | Prediction"

    if "myfile" not in request.files:
        return render_template(
            "prediction.html",
            title=title,
            prediction="Error", probability="0.00",
            second_prediction="No file received", second_probability="0.00",
            third_prediction="Check upload form enctype", third_probability="0.00",
        )

    f = request.files["myfile"]
    if not f or f.filename.strip() == "":
        return render_template(
            "prediction.html",
            title=title,
            prediction="Error", probability="0.00",
            second_prediction="Empty filename", second_probability="0.00",
            third_prediction="Try again", third_probability="0.00",
        )

    suffix = os.path.splitext(f.filename)[1].lower()

    with tempfile.TemporaryDirectory() as tmpdir:
        upload_path = os.path.join(tmpdir, "upload" + suffix)
        f.save(upload_path)

        try:
            wav_path = convert_to_wav_if_needed(upload_path)
            X = extract_mfcc_segments(wav_path)
            p1, s1, p2, s2, p3, s3 = predict_top3(X)

            return render_template(
                "prediction.html",
                title=title,
                prediction=p1, probability=f"{s1 * 100:.2f}",
                second_prediction=p2, second_probability=f"{s2 * 100:.2f}",
                third_prediction=p3, third_probability=f"{s3 * 100:.2f}",
            )

        except subprocess.CalledProcessError:
            return render_template(
                "prediction.html",
                title=title,
                prediction="Error", probability="0.00",
                second_prediction="ffmpeg failed", second_probability="0.00",
                third_prediction="Make sure ffmpeg is on PATH", third_probability="0.00",
            )

        except Exception as e:
            return render_template(
                "prediction.html",
                title=title,
                prediction="Error", probability="0.00",
                second_prediction=type(e).__name__, second_probability="0.00",
                third_prediction=str(e), third_probability="0.00",
            )


@app.route("/project")
def project():
    return render_template("project.html")


@app.route("/about")
def about():
    return render_template("about.html")


@app.route("/contact")
def contact():
    return render_template("contact.html")


if __name__ == "__main__":
    app.run(debug=True)
