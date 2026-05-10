import base64
import copy
import csv
import itertools
import queue
import threading
from io import BytesIO

import cv2 as cv
import mediapipe as mp
import numpy as np
import pyttsx3
from flask import Flask, jsonify, render_template_string, request
from PIL import Image

from model.keypoint_classifier.keypoint_classifier import KeyPointClassifier

app = Flask(__name__)
last_spoken_text = None
tts_queue = queue.Queue()


def tts_worker():
    tts_engine = pyttsx3.init()
    while True:
        text = tts_queue.get()
        if text is None:
            break
        try:
            tts_engine.say(text)
            tts_engine.runAndWait()
        except Exception as e:
            print(f"TTS error: {e}")


tts_thread = threading.Thread(target=tts_worker, daemon=True)
tts_thread.start()


def speak_text(text):
    global last_spoken_text
    if not text or text == last_spoken_text:
        return
    print(f"TTS queued: {text}")
    tts_queue.put(text)
    last_spoken_text = text


def calc_landmark_list(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]
    landmark_point = []
    for landmark in landmarks.landmark:
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)
        landmark_point.append([landmark_x, landmark_y])
    return landmark_point


def pre_process_landmark(landmark_list):
    temp_landmark_list = copy.deepcopy(landmark_list)
    base_x, base_y = 0, 0

    for index, landmark_point in enumerate(temp_landmark_list):
        if index == 0:
            base_x, base_y = landmark_point[0], landmark_point[1]
        temp_landmark_list[index][0] = temp_landmark_list[index][0] - base_x
        temp_landmark_list[index][1] = temp_landmark_list[index][1] - base_y

    temp_landmark_list = list(itertools.chain.from_iterable(temp_landmark_list))
    max_value = max(list(map(abs, temp_landmark_list)))
    if max_value == 0:
        return temp_landmark_list

    return [n / max_value for n in temp_landmark_list]


with open(
    "model/keypoint_classifier/keypoint_classifier_label.csv", encoding="utf-8-sig"
) as f:
    keypoint_classifier_labels = [row[0] for row in csv.reader(f)]

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5,
)
keypoint_classifier = KeyPointClassifier()

HTML_TEMPLATE = """
<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>ASL Recognition Browser App</title>
  <style>
    body { font-family: Arial, sans-serif; margin: 24px; background: #10131a; color: #f1f5f9; }
    .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }
    .card { background: #171c25; border-radius: 12px; padding: 16px; }
    video, img { width: 100%; border-radius: 10px; background: #000; }
    button { margin-top: 12px; padding: 10px 16px; border-radius: 8px; border: none; cursor: pointer; }
    .start { background: #22c55e; color: #052e16; font-weight: 700; }
    .stop { background: #ef4444; color: #fff; font-weight: 700; }
    .pill { display: inline-block; padding: 6px 10px; border-radius: 999px; background: #334155; margin-right: 8px; }
    ul { margin-top: 8px; }
    li { margin-bottom: 4px; }
    .small { color: #94a3b8; font-size: 14px; }
  </style>
</head>
<body>
  <h1>ASL Recognition - Browser Demo</h1>
  <p class="small">Pipeline: Webcam -> MediaPipe (21 landmarks) -> preprocessing -> TFLite classifier -> ASL prediction.</p>

  <div class="grid">
    <div class="card">
      <h3>Live Webcam</h3>
      <video id="video" autoplay playsinline></video><br>
      <button class="start" onclick="startCamera()">Start Camera</button>
      <button class="stop" onclick="stopCamera()">Stop Camera</button>
      <button onclick="enableVoice()">Enable Voice</button>
      <button onclick="testSpeech()">Test Speech</button>
    </div>
    <div class="card">
      <h3>Processed Output</h3>
      <img id="output" alt="Processed frame appears here" />
      <p><span class="pill" id="status">Status: idle</span><span class="pill" id="top">Top: -</span></p>
      <ul id="details"></ul>
    </div>
  </div>
  <div class="card" style="margin-top:20px;">
    <h3>Sentence Builder</h3>
    <p class="small">Build words/sentences from live predicted letters for your final project demo.</p>
    <div id="sentenceBox" style="min-height:48px; background:#0f172a; border-radius:8px; padding:10px; font-size:24px; letter-spacing:1px;">(empty)</div>
    <div style="display:flex; gap:8px; flex-wrap:wrap;">
      <button onclick="addCurrentLetter()">Add Current Letter</button>
      <button onclick="addSpace()">Space</button>
      <button onclick="backspaceSentence()">Backspace</button>
      <button onclick="clearSentence()">Clear</button>
      <button onclick="speakSentence()">Speak Sentence</button>
    </div>
  </div>

  <script>
    const video = document.getElementById("video");
    const output = document.getElementById("output");
    const statusEl = document.getElementById("status");
    const topEl = document.getElementById("top");
    const details = document.getElementById("details");
    const canvas = document.createElement("canvas");
    let stream = null;
    let timer = null;
    let lastSpokenClient = null;
    let voiceEnabled = false;
    let lastSpokenAt = 0;
    let currentPredictedLetter = "";
    let sentenceText = "";

    async function startCamera() {
      if (stream) return;
      stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
      video.srcObject = stream;
      statusEl.textContent = "Status: running";
      timer = setInterval(sendFrame, 300);
    }

    function stopCamera() {
      if (timer) clearInterval(timer);
      timer = null;
      if (stream) {
        stream.getTracks().forEach(track => track.stop());
      }
      stream = null;
      statusEl.textContent = "Status: stopped";
    }

    async function sendFrame() {
      if (!stream || video.videoWidth === 0) return;
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
      const ctx = canvas.getContext("2d");
      ctx.drawImage(video, 0, 0);
      const imageData = canvas.toDataURL("image/jpeg", 0.7);

      const response = await fetch("/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ image: imageData })
      });
      const result = await response.json();

      if (result.image) output.src = result.image;
      topEl.textContent = "Top: " + (result.top_label || "-");
      currentPredictedLetter = getLetterFromTopLabel(result.top_label);
      speakInBrowser(result.top_label);
      details.innerHTML = "";
      (result.summary || []).forEach(line => {
        const li = document.createElement("li");
        li.textContent = line;
        details.appendChild(li);
      });
    }

    function speakInBrowser(topLabel) {
      if (!voiceEnabled) return;
      if (!topLabel) return;
      const label = topLabel.split(" ")[0];
      if (!label || label === lastSpokenClient) return;
      if (!("speechSynthesis" in window)) return;
      const now = Date.now();
      if (now - lastSpokenAt < 1200) return;

      const utterance = new SpeechSynthesisUtterance(label);
      utterance.rate = 0.95;
      utterance.pitch = 1.0;
      window.speechSynthesis.speak(utterance);
      lastSpokenClient = label;
      lastSpokenAt = now;
    }

    function getLetterFromTopLabel(topLabel) {
      if (!topLabel) return "";
      return topLabel.split(" ")[0] || "";
    }

    function renderSentence() {
      const sentenceBox = document.getElementById("sentenceBox");
      sentenceBox.textContent = sentenceText || "(empty)";
    }

    function addCurrentLetter() {
      if (!currentPredictedLetter) return;
      sentenceText += currentPredictedLetter;
      renderSentence();
    }

    function addSpace() {
      if (!sentenceText) return;
      sentenceText += " ";
      renderSentence();
    }

    function backspaceSentence() {
      if (!sentenceText) return;
      sentenceText = sentenceText.slice(0, -1);
      renderSentence();
    }

    function clearSentence() {
      sentenceText = "";
      renderSentence();
    }

    function speakSentence() {
      if (!voiceEnabled || !sentenceText.trim()) return;
      if (!("speechSynthesis" in window)) return;
      const utterance = new SpeechSynthesisUtterance(sentenceText.trim());
      utterance.rate = 0.95;
      utterance.pitch = 1.0;
      window.speechSynthesis.speak(utterance);
    }

    function enableVoice() {
      if (!("speechSynthesis" in window)) {
        alert("Browser speech is not supported in this browser.");
        return;
      }
      voiceEnabled = true;
      const utterance = new SpeechSynthesisUtterance("Voice enabled");
      utterance.rate = 0.95;
      utterance.pitch = 1.0;
      window.speechSynthesis.speak(utterance);
    }

    async function testSpeech() {
      await fetch("/speak_test", { method: "POST" });
    }
  </script>
</body>
</html>
"""


@app.get("/")
def index():
    return render_template_string(HTML_TEMPLATE)


@app.post("/predict")
def predict():
    global last_spoken_text
    print("FRAME LOOP RUNNING", flush=True)
    payload = request.get_json(silent=True) or {}
    image_data = payload.get("image", "")
    if "," not in image_data:
        return jsonify({"image": None, "top_label": None, "summary": []})

    encoded = image_data.split(",", 1)[1]
    frame_bytes = base64.b64decode(encoded)
    pil_image = Image.open(BytesIO(frame_bytes)).convert("RGB")
    frame = np.array(pil_image)

    results = hands.process(frame)
    annotated = frame.copy()

    top_label = None
    predicted_text = None
    top_conf = 0.0
    summary = []

    if results.multi_hand_landmarks:
        for idx, (hand_landmarks, handedness) in enumerate(
            zip(results.multi_hand_landmarks, results.multi_handedness), start=1
        ):
            landmark_list = calc_landmark_list(annotated, hand_landmarks)
            pre_processed_landmark_list = pre_process_landmark(landmark_list)
            class_id, scores = keypoint_classifier.predict(pre_processed_landmark_list)

            probs = np.exp(scores - np.max(scores))
            probs = probs / np.sum(probs)
            confidence = float(probs[class_id])
            label = keypoint_classifier_labels[class_id]

            if confidence > top_conf:
                top_conf = confidence
                top_label = f"{label} ({confidence:.1%})"
                predicted_text = label

            summary.append(
                f"Hand {idx} [{handedness.classification[0].label}]: {label} ({confidence:.1%})"
            )

            mp_drawing.draw_landmarks(
                annotated,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style(),
            )
    else:
        print("NO HAND DETECTED", flush=True)

    # Debug + speak trigger exactly in frame prediction flow.
    print("PREDICTED TEXT:", predicted_text, flush=True)
    if predicted_text and predicted_text != last_spoken_text:
        print("TTS CALLED:", predicted_text, flush=True)
        speak_text(predicted_text)
        last_spoken_text = predicted_text

    success, jpg = cv.imencode(".jpg", cv.cvtColor(annotated, cv.COLOR_RGB2BGR))
    out_img = (
        "data:image/jpeg;base64," + base64.b64encode(jpg.tobytes()).decode("utf-8")
        if success
        else None
    )
    return jsonify({"image": out_img, "top_label": top_label, "summary": summary})


@app.post("/speak_test")
def speak_test():
    speak_text("Text to speech is working")
    return jsonify({"ok": True})


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
