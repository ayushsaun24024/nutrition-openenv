import requests
import os
from openai import OpenAI

API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")
API_KEY = os.getenv("HF_TOKEN") or os.getenv("OPENAI_API_KEY")

ENV_URL = os.getenv("ENV_URL", "https://ayushsaun-nutrition-openenv.hf.space")

TASK_NAME = "hard"
BENCHMARK = "nutrition-env"

MAX_STEPS = 10
SUCCESS_SCORE_THRESHOLD = 0.7

client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)


def log_start():
    print(f"[START] task={TASK_NAME} env={BENCHMARK} model={MODEL_NAME}", flush=True)


def log_step(step, action, reward, done, error):
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}", flush=True)


def log_end(success, steps, score, rewards):
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}", flush=True)


def get_action_from_model(obs, prev_reward=None, reward_trend=None):
    prompt = f"""
You are an intelligent nutritional decision-making agent.

Objective:
Make balanced dietary decisions over time, not just immediate gains.

Goals:
- Reach calorie target efficiently
- Avoid overshooting calories
- Maintain balanced nutrition (protein, carbs, fat)
- Avoid excessive sugar and fat
- Avoid repeating the same food excessively
- Avoid over-reliance on a single nutrient source (e.g., only protein)

Current state:
- Current step: {obs.get('step', '?')} / {MAX_STEPS}
- Remaining steps: {MAX_STEPS - obs.get('step', 0)}
- Calories consumed: {obs.get('calories_consumed', 0)}
- Target calories: {obs.get('calorie_target', 2000)}
- Hunger: {obs.get('hunger', 0.5)}
- Previous reward: {prev_reward if prev_reward is not None else "N/A"}
- Reward trend: {reward_trend if reward_trend else "N/A"}

Available actions:
apple, rice, chicken, snack, milk, egg, skip

Important behavioral rules:
- Do NOT repeat the same food too many times in a row
- If a food has been used repeatedly, choose a different one
- After sufficient protein intake, shift towards balanced foods
- Think ahead to avoid penalties from imbalance
- If reward decreased recently, try a different strategy
- keep it as a mix of fruits, veges, rice and chicken not concentrated towards one
- cnosider fibres, protiens, vitamic (vitamic - c) and all make sure it is balanced diet

Respond with ONLY one word:
apple OR rice OR chicken OR snack OR milk OR egg OR skip
"""

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": "You are a precise decision-making agent."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
            max_tokens=10,
        )

        action = response.choices[0].message.content.strip().lower()

        if action not in ["apple", "rice", "chicken", "snack", "milk", "egg", "skip"]:
            return "rice"

        return action

    except Exception as e:
        print(f"[DEBUG] OpenAI error: {e}", flush=True)
        return "rice"


def main():
    rewards = []
    steps_taken = 0
    success = False
    score = 0.0

    log_start()

    prev_reward = None
    prev_prev_reward = None

    try:
        res = requests.post(f"{ENV_URL}/reset", json={"task": TASK_NAME}, timeout=30).json()
        obs = res.get("observation", {})

        for step in range(1, MAX_STEPS + 1):
            if prev_reward is None or prev_prev_reward is None:
                trend = "N/A"
            else:
                trend = "increasing" if prev_reward > prev_prev_reward else (
                    "decreasing" if prev_reward < prev_prev_reward else "stable"
                )

            action = get_action_from_model(obs, prev_reward=prev_reward, reward_trend=trend)

            res = requests.post(f"{ENV_URL}/step", json={"food": action}, timeout=30).json()

            obs = res.get("observation", {})
            reward = res.get("reward", 0.0)
            done = res.get("done", False)
            error = res.get("info", {}).get("error")

            rewards.append(reward)
            steps_taken = step

            log_step(step, action, reward, done, error)

            prev_prev_reward = prev_reward
            prev_reward = reward

            if done:
                break

        if rewards:
            score = sum(rewards) / len(rewards)
        score = max(0.0, min(score, 1.0))
        success = score >= SUCCESS_SCORE_THRESHOLD

    except Exception as e:
        print(f"[DEBUG] Fatal error: {e}", flush=True)

    finally:
        log_end(success, steps_taken, score, rewards)


if __name__ == "__main__":
    main()
