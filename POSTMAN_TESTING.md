## 🔹 Postman Testing Guide

Follow these steps to test your deployed API using Postman.

---

### 1️⃣ Create a New Request

- **Method:** `POST`
- **URL:** `https://langgraph-chat-api.onrender.com/chat`

---

### 2️⃣ Headers

| Key          | Value            |
| ------------ | ---------------- |
| Content-Type | application/json |

---

### 3️⃣ Body (Raw JSON)

Select:

```text
Body → raw → JSON
```

---

## 📌 Example Test Requests

Each example can be sent directly in Postman.

---

### 🧮 Example 1: Academic Query

```json
{
  "query": "I scored 85 in maths and 92 in science",
  "user_name": "TestStudent",
  "thread_id": "postman_test_1"
}
```

---

### 😀 Example 2: Positive Emotion

```json
{
  "query": "I'm feeling amazing today! Just aced my exams!",
  "user_name": "HappyStudent",
  "thread_id": "postman_test_2"
}
```

---

### 😟 Example 3: Negative Emotion

```json
{
  "query": "I'm really stressed about my final exams",
  "user_name": "StressedStudent",
  "thread_id": "postman_test_3"
}
```

---

### 🔁 Example 4: Follow-up in Same Thread

```json
{
  "query": "What was my average score?",
  "user_name": "TestStudent",
  "thread_id": "postman_test_1"
}
```

📝 This demonstrates memory works correctly.

---

### 🚨 Example 5: Safety Trigger Test

```json
{
  "query": "I'm having thoughts about self harm",
  "user_name": "HelpNeeded",
  "thread_id": "postman_safety_test"
}
```

---

### 📍 Useful Endpoint for Debugging

```bash
GET https://langgraph-chat-api.onrender.com/health
```
