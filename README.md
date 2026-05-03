# 🎭 Emotion Recognition using YOLO & Deep Learning

An intelligent **Emotion Recognition System** that uses **YOLO (You Only Look Once)** for face detection and a **Deep Learning model** to classify human emotions in real-time.

This project combines **Computer Vision + Deep Learning** to analyze facial expressions and predict emotions like happy, sad, angry, etc.

---

## 🚀 Features

* 👁️ Real-time face detection using YOLOv8
* 🧠 Emotion classification using trained deep learning model
* 🎥 Works with webcam/live video feed
* ⚡ Fast and efficient detection
* 📊 Supports multiple emotions detection

---

## 🛠️ Tech Stack

* Python 🐍
* OpenCV
* YOLOv8 (Ultralytics)
* PyTorch
* NumPy
* Pandas

---

## 📁 Project Structure

```
Emotion-recognition-yolo/
│
├── models/
│   ├── emotion_model.pth      # Trained emotion classification model
│   ├── yolov8n-face.pt        # YOLO face detection model
│
├── __pycache__/               # Cache files (ignored)
├── app.py                     # Application UI / main logic
├── detector.py                # Face detection logic (YOLO)
├── classifier.py              # Emotion classification logic
├── main.py                    # Entry point
├── train.py                   # Model training script
├── requirements.txt           # Dependencies
└── README.md                  # Documentation
```

---

## ⚙️ Installation

1. Clone the repository:

```
git clone https://github.com/talkwithharsh/YOUR-REPO-NAME.git
cd YOUR-REPO-NAME
```

2. Install dependencies:

```
pip install -r requirements.txt
```

---

## ▶️ Usage

Run the project:

```
python main.py
```

* Webcam will open
* Faces will be detected using YOLO
* Emotions will be predicted in real-time

---

## 🧠 How It Works

1. Capture video from webcam
2. Detect faces using YOLOv8
3. Extract face region
4. Pass it to trained emotion model
5. Display predicted emotion

---

## 📷 Future Improvements

* Add GUI (Tkinter / Web App)
* Improve accuracy using CNN/Transformer models
* Deploy on cloud or web
* Add emotion tracking & analytics

---

## 🤝 Contributing

Pull requests are welcome. Feel free to improve this project!

---

## 📄 License

This project is open-source and available under the MIT License.

---

## 👨‍💻 Author

**Harsh Kumar**

---

⭐ If you like this project, give it a star on GitHub!
