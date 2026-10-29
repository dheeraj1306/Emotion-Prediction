# 🎭 Emotion Prediction — Deep Learning NLP Web App

An end-to-end machine learning project that detects the **emotion behind a sentence** using a **Bidirectional GRU** neural network, served through a **FastAPI** backend and a clean **HTML/CSS/JavaScript** frontend.

Type something like *"I feel so alone and hopeless today"* and the app returns the predicted emotion (😢 sadness), a confidence score, and the full probability breakdown across all six emotions.

<!-- Add a screenshot or GIF of your app here. This is the single most useful thing for recruiters. -->
<!-- ![App Demo](static/demo.png) -->

---

## ✨ Features

- Classifies text into **6 emotions**: sadness 😢, joy 😄, love ❤️, anger 😠, fear 😨, surprise 😲
- Compares **Simple RNN vs LSTM vs GRU**, then improves the best one into a **stacked Bidirectional GRU**
- Handles **class imbalance** using balanced class weights
- **REST API** built with FastAPI, with automatic Swagger docs at `/docs`
- Input validation using **Pydantic** (1–2000 characters)
- Model and tokenizer loaded **once at server startup** (FastAPI lifespan) for fast predictions
- Health-check endpoint for monitoring
- Responsive frontend served directly by FastAPI

---

## 🛠️ Tech Stack

| Layer | Technologies |
|---|---|
| **Machine Learning** | Python, TensorFlow / Keras, scikit-learn, NumPy, Pandas |
| **Data & Visualization** | Hugging Face Datasets, Matplotlib, Seaborn |
| **Backend** | FastAPI, Uvicorn, Pydantic |
| **Frontend** | HTML, CSS, JavaScript |

---

## 📊 Dataset

- **Source:** [`dair-ai/emotion`](https://huggingface.co/datasets/dair-ai/emotion) on Hugging Face
- English Twitter messages labelled with one of six emotions
- The class distribution is **imbalanced**, so balanced class weights are used during training

---

## 🧠 Model Development

All experiments are in [`final_clean.ipynb`](final_clean.ipynb).

**1. Preprocessing**
- Keras `Tokenizer` with a 10,000-word vocabulary and an `<unk>` token for unseen words
- Sequences padded/truncated to a fixed length of **50**

**2. Phase 1 — Baseline comparison**

| Model | Architecture |
|---|---|
| Simple RNN | Embedding(128) → RNN(128) → Dropout → RNN(64) → Dropout → Softmax |
| LSTM | Embedding(128) → LSTM(128) → Dropout → LSTM(64) → Dropout → Softmax |
| GRU | Embedding(128) → GRU(128) → Dropout → GRU(64) → Dropout → Softmax |

**3. Phase 2 — Final model (Bidirectional GRU)**

```
Embedding (300-dim)
   → Bidirectional GRU (128, return_sequences)
   → Dropout (0.5)
   → Bidirectional GRU (64)
   → Dropout (0.5)
   → Dense (6, softmax)
```

**Training setup:** Adam optimizer, sparse categorical cross-entropy, batch size 32, up to 20 epochs, `EarlyStopping` (patience 3, restores best weights).

### Results

| Model | Test Accuracy |
|---|---|
| Simple RNN | 26.4% |
| LSTM | 33.9% |
| GRU | 28.9% |
| **Bidirectional GRU (final)** | **92.1%** |


---

## 🏗️ Architecture

```
┌──────────────┐   POST /predict    ┌─────────────────────────────┐
│   Frontend   │ ─────────────────▶ │          FastAPI            │
│ HTML/CSS/JS  │                    │  1. Clean text              │
│              │ ◀───────────────── │  2. Tokenize + pad          │
└──────────────┘   JSON response    │  3. BiGRU prediction        │
                                    │  4. Return emotion + scores │
                                    └─────────────────────────────┘
```

---

## 📁 Project Structure

```
Emotion-Prediction/
├── Artifacts/            # Trained model (.keras) and tokenizer (.pkl)
├── static/               # Frontend (index.html, CSS, JavaScript)
├── final_clean.ipynb     # Data exploration, training, evaluation
├── main.py               # FastAPI application
├── requirements.txt      # Python dependencies
└── .gitignore
```

---

## 🚀 Getting Started

### Prerequisites
- Python 3.10 or 3.11

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/dheeraj1306/Emotion-Prediction.git
cd Emotion-Prediction

# 2. Create and activate a virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the server
uvicorn main:app --reload
```

Then open:
- **App UI:** http://127.0.0.1:8000
- **API docs (Swagger):** http://127.0.0.1:8000/docs

---

## 🔌 API Reference

### `GET /health`
Checks that the server is up and the model is loaded.

```json
{ "status": "Server is running", "model_loaded": true }
```

### `POST /predict`
Predicts the emotion of a sentence.

**Request**
```json
{ "text": "I feel so happy and excited" }
```

**Response**
```json
{
  "text": "I feel so happy and excited",
  "predicted_emotion": "joy",
  "confidence": 0.93,
  "all_probabilites": {
    "sadness": 0.01,
    "joy": 0.93,
    "love": 0.02,
    "anger": 0.01,
    "fear": 0.01,
    "surprise": 0.02
  }
}
```

**cURL example**
```bash
curl -X POST "http://127.0.0.1:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{"text": "I am furious that they cancelled the trip"}'
```

---

## 🔮 Future Improvements

- Use pre-trained embeddings (GloVe) or a Transformer model (BERT / DistilBERT)
- Add a validation split separate from the test set
- Improve performance on under-represented emotions (e.g. surprise, love)
- Dockerize and deploy to Render / Railway / Hugging Face Spaces
- Add unit tests for the API

---

## 👤 Author

**Dheeraj**
- GitHub: [@dheeraj1306](https://github.com/dheeraj1306)

⭐ If you found this project useful, consider giving it a star!
