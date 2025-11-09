# SmartVision

SmartVision is a full-stack deep learning application that automates image classification using a Convolutional Neural Network (CNN). The system enables users to upload an image via a React-based web interface, processes it in a Flask backend, and returns the predicted label and confidence score in real time.

Built with TensorFlow, Flask, and React, this project demonstrates scalable ETL (Extract, Transform, Load) workflows for image data, real-time inference, and interactive visualization aligning closely with real-world AI engineering and data pipeline development.

🧩 Key Features

🎯 Real-time image classification with CNN

⚙️ Flask REST API for prediction endpoint

🧠 TensorFlow-based model with dynamic preprocessing

💾 Efficient ETL data pipeline (data → preprocessing → model → output)

💡 Interactive React dashboard for predictions

🔍 Model explainability via confidence visualization

🧠 Tech Stack

Frontend: React.js, Axios, HTML5, CSS3
Backend: Flask, Flask-CORS, TensorFlow, Pillow, NumPy
Database (optional): MongoDB or SQLite (for logging results)
Tools: VS Code, Postman, Git, Google Colab

⚙️ Architecture Diagram
            ┌────────────────────────┐
            │   React Frontend (UI)  │
            │  • Image Upload        │
            │  • Display Results     │
            └──────────┬─────────────┘
                       │  HTTP POST /predict
                       ▼
            ┌────────────────────────┐
            │   Flask Backend (API)  │
            │  • Preprocess image    │
            │  • Model inference     │
            │  • JSON response       │
            └──────────┬─────────────┘
                       │
                       ▼
            ┌────────────────────────┐
            │   TensorFlow Model     │
            │  • CNN architecture    │
            │  • Trained on dataset  │
            └────────────────────────┘
Setup & Execution

🔹 Backend Setup
cd smartvision/backend
python -m venv venv
source venv/bin/activate   # (Windows: venv\Scripts\activate)
pip install -r requirements.txt
python train.py            # trains CNN model
python app.py              # starts API at http://localhost:5000

🔹 Frontend Setup

cd smartvision/frontend
npm install
npm start

Visit 👉 http://localhost:3000

Upload an image → View predicted class and confidence

🧩 API Reference

Endpoint: /predict
Method: POST
Body: multipart/form-data (key: file)
Response:
{
  "class": "leaf_disease",
  "confidence": 0.95
}

🌟 Key Learnings & Impact

Designed a production-style data preprocessing pipeline for image ingestion.

Implemented scalable model inference architecture combining Flask REST APIs and TensorFlow.

Learned deployment principles relevant to AI-powered web apps.

🔮 Future Improvements

Integrate model explainability with Grad-CAM visualizations.

Enable cloud storage (AWS S3) for image logs.

Containerize with Docker for multi-service deployment.

Hosting Options: Render / AWS EC2 / ngrok for testing
