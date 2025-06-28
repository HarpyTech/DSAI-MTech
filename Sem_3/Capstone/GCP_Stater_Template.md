# 🧠 Building a Custom Agentic AI System on GCP using Google AI Studio

This blog post walks through how to create a **custom Agentic AI system** using your own machine learning models, deployed on **Google Cloud Platform (GCP)** and orchestrated with **Gemini 1.5 Pro via Google AI Studio**. You'll learn how to expose your models as APIs (via Cloud Functions or Vertex AI Endpoints) and how to use Gemini's function calling capabilities to plan and execute workflows across these models.

---

## 🔧 Step 1: Expose Your Models as APIs

To integrate with Gemini, your models need to be callable via HTTP. You can achieve this in two main ways:

### ✅ Option A: Use Cloud Functions (Fastest for Prototyping)

Wrap your model with Flask and deploy as a Google Cloud Function. This works great for smaller models and quick iteration.

### ✅ Option B: Use Vertex AI Model Endpoints (Better for Production)

Deploy your trained model as a containerized service to **Vertex AI**, making it highly scalable and monitorable.

### 📌 Example: Sentiment Analysis as Cloud Function

```python
# main.py
from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)
model = joblib.load("sentiment_model.pkl")

@app.route("/", methods=["POST"])
def predict():
    data = request.get_json()
    text = data.get("text", "")
    prediction = model.predict([text])[0]
    return jsonify({"sentiment": prediction})
```

**requirements.txt**:
```
flask
scikit-learn
```

Deploy it:
```bash
gcloud functions deploy sentimentTool \
  --runtime python310 \
  --trigger-http \
  --allow-unauthenticated
```

---

### 📦 Deploy to Vertex AI with Docker

For production-grade performance, containerize your model and deploy to Vertex AI.

#### Dockerfile Example
```Dockerfile
FROM python:3.10-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
EXPOSE 8080
CMD ["python", "main.py"]
```

**main.py (FastAPI or Flask)**
```python
from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)
model = joblib.load("model.pkl")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    input_data = data.get("text", "")
    result = model.predict([input_data])[0]
    return jsonify({"result": result})
```

Build and push to Artifact Registry:
```bash
gcloud builds submit --tag gcr.io/YOUR_PROJECT_ID/sentiment-model
```

Deploy to Vertex AI:
```bash
gcloud ai models upload --region=us-central1 \
  --display-name="sentiment-model" \
  --container-image-uri=gcr.io/YOUR_PROJECT_ID/sentiment-model

gcloud ai endpoints create --region=us-central1 --display-name="sentiment-endpoint"
gcloud ai endpoints deploy-model --region=us-central1 \
  --model=MODEL_ID --display-name="sentiment-deployed" --endpoint=ENDPOINT_ID
```

---

## 🧠 Step 2: Define Tool Schema for Gemini (Function Calling)

Gemini supports calling external functions via JSON schema. You register these in **Google AI Studio** to let Gemini know which tools it can access.

### 🧩 Sample Schema: `gemini_tool_schema.json`
```json
{
  "functions": [
    {
      "name": "classify_image",
      "description": "Classifies an image from a Cloud Storage URL.",
      "parameters": {
        "type": "object",
        "properties": {
          "image_url": { "type": "string" }
        },
        "required": ["image_url"]
      }
    },
    {
      "name": "analyze_sentiment",
      "description": "Analyzes sentiment from a review.",
      "parameters": {
        "type": "object",
        "properties": {
          "text": { "type": "string" }
        },
        "required": ["text"]
      }
    },
    {
      "name": "summarize_text",
      "description": "Summarizes review text.",
      "parameters": {
        "type": "object",
        "properties": {
          "text": { "type": "string" }
        },
        "required": ["text"]
      }
    },
    {
      "name": "predict_regression",
      "description": "Predicts numerical score using regression model.",
      "parameters": {
        "type": "object",
        "properties": {
          "features": {
            "type": "array",
            "items": { "type": "number" }
          }
        },
        "required": ["features"]
      }
    }
  ]
}
```

Register this JSON in your **Gemini AI Studio** configuration.

---

## 🔁 Step 3: Add More Tools (e.g., Regression Model)

Here’s an example of how to expose a regression model.

```python
# regression_main.py
from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)
model = joblib.load("regression_model.pkl")

@app.route("/", methods=["POST"])
def predict():
    data = request.get_json()
    features = np.array(data.get("features", [])).reshape(1, -1)
    prediction = model.predict(features)[0]
    return jsonify({"prediction": float(prediction)})
```

You can either deploy this to Cloud Functions or containerize it for Vertex AI, just like the sentiment model.

---

## 💬 Step 4: Gemini Prompt to Trigger Reasoning

In Gemini AI Studio, provide a prompt that lets the model understand when and how to use your tools.

### 🧠 Example Prompt
```
You are an intelligent assistant with access to the following tools:
- classify_image(image_url)
- analyze_sentiment(text)
- summarize_text(text)
- predict_regression(features)

Given user input, determine which tool(s) to call, in which order, and summarize the results for the user.

Input: "Here's my review: 'Terrible experience. Phone broke in a week.'"
```

Gemini will then select and call the appropriate tool(s) using its internal planner.

---

## ✅ Conclusion

You’ve now built the core of an Agentic AI system using Google Cloud and Gemini. The Gemini LLM coordinates task execution across your models based on context and goal.

### 🛠 Recap:
- Expose ML models as REST APIs using Cloud Functions or Vertex AI Endpoints
- Register them as tools using Gemini function-calling schemas
- Prompt Gemini to act as a controller agent

### 📈 Next Steps:
- Add memory (via Firestore or BigQuery)
- Integrate with Streamlit or Firebase for UI
- Use Google Workflows to control multi-step logic (outside of Gemini if needed)

---

