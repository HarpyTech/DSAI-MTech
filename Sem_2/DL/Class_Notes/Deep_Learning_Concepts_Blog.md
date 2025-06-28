# 🧠 Deep Learning Concepts: A Compact Guide with Visuals & Code

This post explores key deep learning concepts with **diagrams first**, followed by **brief explanations** and **code snippets** to help you understand how they work in practice.

---

## 🔁 Neural Network Process

![Neural Process](blog_assets/neural_process.png)

A neural network processes input through layers:

```python
# Pseudo process
input -> Dense -> Activation -> ... -> Output
loss = loss_fn(y_pred, y_true)
loss.backward()  # backpropagation
optimizer.step()
```

---

## ⚙️ Batch Normalization

![Batch Norm](blog_assets/batch_norm.png)

**Batch Normalization** normalizes inputs in a batch. It improves speed and stability during training.

```python
from tensorflow.keras.layers import BatchNormalization
model.add(BatchNormalization())
```

---

## 📉 Gradient Descent Variants

**Gradient Descent** optimizes weights by minimizing a loss function.

```python
# Stochastic Gradient Descent
weight = weight - learning_rate * gradient
```

### Types:
- **Batch GD**: Entire dataset
- **Stochastic GD**: One data point
- **Mini-batch GD**: Small batches (most common)

---

## 🚀 Activation Functions – ReLU

![ReLU](blog_assets/relu.png)

```python
def relu(x):
    return max(0, x)
```

Used to introduce non-linearity. Fast and widely used, though it may suffer from "dying ReLU" issue.

---

## 🎯 YOLO Algorithm - Object Detection

![YOLO](blog_assets/yolo.png)

YOLO (You Only Look Once) is a fast, real-time object detection system.

```python
# Conceptual steps
1. Divide image into grid
2. Predict bounding boxes and class probs
3. Apply confidence threshold
4. Perform Non-Maximum Suppression
```

---

## 🎓 Additional Topics

### Causes of Overfitting
- Too complex model
- Limited data
- Inadequate regularization

**Fixes**: Dropout, Early Stopping

### CNN Layers
- **Convolution**: Extract features
- **Pooling**: Reduce spatial size
- **Dense**: Final classifier

### Vanishing Gradient
Gradients diminish during backpropagation.

**Solutions**: ReLU, BatchNorm, Skip Connections (ResNet)

---

## 📊 Object Detection Metrics
- **IoU**, **Precision**, **Recall**, **F1-Score**, **mAP**

**NMS (Non-Max Suppression)**: Removes overlapping detections.

---

## 🧮 Common Activation Functions

```python
# Tanh
def tanh(x):
    return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))

# Softmax
def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)
```

---

> This guide is ideal for interviews and exams. Let me know if you’d like this exported as an HTML or Jupyter Notebook!

