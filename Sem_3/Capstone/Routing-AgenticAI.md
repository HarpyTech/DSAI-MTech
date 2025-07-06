Here’s how prompt execution would look like for the **Heart Attack Risk Predictor** and **Diabetic Risk Classifier** using your defined agentic tools:

---

### 🔍 Example 1: Heart Attack Risk Prediction Prompt

```python
patient_info = """
Age: 58
Gender: Male
Chest Pain Type: Typical Angina
Resting Blood Pressure: 140 mm Hg
Serum Cholesterol: 260 mg/dl
Fasting Blood Sugar > 120 mg/dl: Yes
Resting ECG Results: ST-T wave abnormality
Max Heart Rate Achieved: 150 bpm
Exercise Induced Angina: Yes
ST Depression: 2.3
Slope of ST Segment: Downsloping
Number of Major Vessels Colored by Fluoroscopy: 1
Thalassemia: Fixed Defect
"""

response = predict_heart_attack_risk.run(patient_info)
print(response)
```

✅ **This will trigger Gemini Pro** to respond with something like:

> **Risk Level:** High
> **Reasoning:** The patient is a 58-year-old male with chest pain, elevated blood pressure, abnormal ECG, and exercise-induced angina. These are strong indicators of coronary artery disease and elevated heart attack risk.

---

### 🔍 Example 2: Diabetic Classification Prompt

```python
patient_metrics = """
Age: 42
BMI: 31.2
Fasting Blood Sugar: 110 mg/dl
Insulin Level: 18 µU/mL
Family History: Yes
Autoantibodies: Negative
Physical Activity: Low
Dietary Habits: High sugar intake
Genetic Marker: Positive for TCF7L2 mutation
"""

response = classify_diabetes.run(patient_metrics)
print(response)
```

✅ **Gemini Pro response might be:**

> **Diagnosis:** Type 2 Diabetes
> **Rationale:** Elevated BMI, fasting blood sugar, positive genetic marker, low physical activity, and poor dietary habits suggest insulin resistance commonly associated with Type 2 Diabetes.

---

Would you like these two examples baked directly into your script for testing or auto-evaluation?


Great question! In an **Agentic AI architecture**, when an end user provides a **free-text paragraph** (e.g., *“My father has high cholesterol, experiences chest pain and shortness of breath during mild activity”*), the model needs to:

---

### ✅ **Step-by-Step Processing Pipeline**

#### 🔹 1. **Intent Classification & Entity Extraction**

Use a **lightweight intent extractor** (can be LLM or rules) to understand what the user is talking about:

```python
def identify_task_and_entities(text: str) -> str:
    prompt = f"""
    You are a medical assistant. Based on this paragraph, decide whether the task is:
    - Heart Risk Prediction
    - Diabetes Classification
    - Clinical Treatment Summary

    Input:
    {text}

    Respond ONLY with the name of the task and list the key features present.
    """
    model = GenerativeModel("gemini-pro")
    return model.generate_content(prompt).text
```

**Example Output:**

```text
Task: Heart Risk Prediction  
Features: Chest Pain, Shortness of Breath, High Cholesterol, Age: 60
```

---

#### 🔹 2. **Route to Appropriate Tool**

You can now route the request like this:

```python
result = identify_task_and_entities(user_input)

if "Heart Risk" in result:
    response = predict_heart_attack_risk.run(user_input)
elif "Diabetes" in result:
    response = classify_diabetes.run(user_input)
elif "Treatment Summary" in result:
    response = summarize_treatment.run(user_input)
```

You can even combine this logic inside a **custom routing agent**, or dynamically call `agent.run()` with the parsed action hint:

```python
agent.run(f"Predict using appropriate model:\n{user_input}")
```

LangChain’s `ZeroShotAgent` is capable of selecting the correct tool based on the tool description and user query if tools are well-labeled.

---

### 🧠 Smart Tool Descriptions Help

To help LangChain’s agent decide correctly, use **descriptive tool names**:

```python
Tool(
    name="predict_heart_attack_risk",
    func=predict_heart_attack_risk,
    description="Predict heart attack risk level based on symptoms like chest pain, cholesterol, ECG, etc."
),
Tool(
    name="classify_diabetes",
    func=classify_diabetes,
    description="Identify the type of diabetes based on patient metrics like insulin, glucose, BMI, genetics."
)
```

---

### 🚀 Want to Add This to Your Code?

I can:

* Inject `identify_task_and_entities` as a tool or standalone
* Extend the agent logic with auto-routing
* Wrap the logic inside a single inference endpoint (Vertex AI or Cloud Function)

Let me know which way you prefer to proceed.
