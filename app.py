import cv2
import time
import threading
from flask import Flask, render_template_string, Response, jsonify
from detector import FaceDetector
from classifier import EmotionClassifier

app = Flask(__name__)

# ------------------- Load Models -------------------
detector = FaceDetector("models/yolov8n-face.pt")
clf = EmotionClassifier(model_path="models/emotion_model.pth")

# ------------------- Global State -------------------
emotion_counts = {}
emotion_times = {}
current_emotion = None
emotion_start_time = None
session_start = time.time()

# ------------------- Camera Capture -------------------
def gen_frames():
    global current_emotion, emotion_start_time

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open camera")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        faces = detector.detect(frame)

        for (x1,y1,x2,y2) in faces:
            face = frame[y1:y2, x1:x2]
            if face.size == 0: 
                continue

            label, prob = clf.predict(face)

            # Update emotion counts
            emotion_counts[label] = emotion_counts.get(label, 0) + 1

            # Track emotion time
            global emotion_times
            if current_emotion != label:
                # Save time for last emotion
                if current_emotion and emotion_start_time:
                    elapsed = time.time() - emotion_start_time
                    emotion_times[current_emotion] = emotion_times.get(current_emotion, 0) + elapsed

                current_emotion = label
                emotion_start_time = time.time()

            # Draw UI on frame
            cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
            cv2.putText(frame, f"{label} {prob:.2f}", (x1,y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)

        # Encode frame
        _, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()

# ------------------- Routes -------------------
@app.route('/')
def index():
    # HTML with dark theme + session stats
    return render_template_string('''
    <html>
    <head>
        <title>Emotion Recognition</title>
        <style>
            body { background-color: #000; color: #fff; font-family: Arial, sans-serif; text-align: center; }
            .container { display: flex; flex-direction: column; align-items: center; }
            .stats { margin-top: 20px; width: 60%; background: #111; padding: 20px; border-radius: 12px; }
            video, img { border-radius: 12px; box-shadow: 0px 0px 15px #0f0; }
            h1 { color: #0f0; }
        </style>
        <script>
            function updateStats() {
                fetch('/stats')
                    .then(r => r.json())
                    .then(data => {
                        document.getElementById("session").innerText = data.session_time + "s";
                        document.getElementById("counts").innerText = JSON.stringify(data.emotion_counts, null, 2);
                        document.getElementById("times").innerText = JSON.stringify(data.emotion_times, null, 2);
                    });
            }
            setInterval(updateStats, 1000);
        </script>
    </head>
    <body>
        <div class="container">
            <h1>YOLO + Emotion Recognition</h1>
            <img src="{{ url_for('video_feed') }}" width="640">
            <div class="stats">
                <h2>Session Time: <span id="session">0</span></h2>
                <h3>Emotion Counts:</h3>
                <pre id="counts">{}</pre>
                <h3>Per Emotion Time (s):</h3>
                <pre id="times">{}</pre>
            </div>
        </div>
    </body>
    </html>
    ''')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/stats')
def stats():
    # Calculate ongoing emotion time
    global emotion_times, current_emotion, emotion_start_time
    if current_emotion and emotion_start_time:
        elapsed = time.time() - emotion_start_time
        emotion_times[current_emotion] = emotion_times.get(current_emotion, 0) + elapsed
        emotion_start_time = time.time()  # reset timer

    return jsonify({
        "session_time": int(time.time() - session_start),
        "emotion_counts": emotion_counts,
        "emotion_times": {k: round(v, 2) for k,v in emotion_times.items()}
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
