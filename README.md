# 🍽️ Nutrition OpenEnv Environment

## 📌 Overview

This project implements a **real-world reinforcement learning environment** for adaptive nutritional decision-making.

The goal is to simulate how individuals make food choices under constraints such as calorie targets, hunger, and step limits.

Agents interact with the environment via:

- `reset()`
- `step(action)`
- `state()`

and learn to optimize dietary decisions over time.

---

## 🎯 Objective

The agent must:

- Reach a **daily calorie target**
- Avoid **overeating**
- Avoid **undereating**
- Make **efficient decisions** within limited steps

---

## 🧠 Environment Design

### 🔹 Observation Space

| Field | Type | Description |
|------|------|------------|
| calories_consumed | float | Total calories consumed so far |
| calorie_target | float | Target daily calories |
| hunger | float | Estimated hunger (0–1) |
| step | int | Current timestep |

---

### 🔹 Action Space

Agent chooses one:

- `apple` → 95 calories  
- `rice` → 200 calories  
- `chicken` → 250 calories  
- `snack` → 300 calories  
- `skip` → 0 calories  

---

### 🔹 Reward Function

Reward is continuous and shaped:

- Closer to target → higher reward  
- Overshooting → penalty  
- Undereating → lower reward  

Formula (simplified):

```Bash
reward = 1 - (|target - consumed| / target)
```


Penalty applied if exceeding target.

---

## 🧪 Tasks & Evaluation

### 🟢 Easy Task
- Reach calorie target
- Focus: basic correctness

---

### 🟡 Medium Task
- Reach target efficiently
- Penalizes unnecessary steps

---

### 🔴 Hard Task
- Precise calorie matching
- Strong penalty for overshooting

---

### 📊 Scoring

All tasks produce:
```Bash
score ∈ [0.0, 1.0]
```


Based on:
- accuracy
- efficiency
- safety (no overshoot)

---

## ⚙️ API Endpoints

| Endpoint | Method | Description |
|--------|--------|------------|
| `/reset` | POST | Reset environment |
| `/step` | POST | Take action |
| `/state` | GET | Get internal state |

---

## 🚀 Running Locally

```bash
pip install -r requirements.txt
python -m uvicorn server.app:app --reload
```

Open:
```Bash
http://127.0.0.1:8000/docs
```

## Inference

### Run:
```Bash
python inference.py
```

### Uses:

OpenAI API (gpt-4o-mini)
environment interaction loop
structured logging

## 🔐 Environment Variables

Set before running inference:

OPENAI_API_KEY=your_key_here

## 🐳 Deployment

This project is containerized using Docker and deployable on:

Hugging Face Spaces (Docker mode)

## 📦 Project Structure
```Bash
my_env/
    env.py
    models.py
    tasks.py

server/
    app.py

inference.py
openenv.yaml
Dockerfile
requirements.txt
pyproject.toml
```

## 💡 Key Features
Real-world decision modeling
Deterministic environment
Multi-task evaluation
Continuous reward shaping
OpenEnv compliant
Lightweight (CPU-friendly)

### 🧑‍💻 Author

Ayush