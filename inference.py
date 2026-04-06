import requests
import os
from openai import OpenAI

API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")
API_KEY = os.getenv("HF_TOKEN") or os.getenv("OPENAI_API_KEY")

ENV_URL = os.getenv("ENV_URL", "http://127.0.0.1:8000")

TASK_NAME = "nutrition-task"
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


def get_action_from_model(obs):
    prompt = f"""
You are controlling a nutrition agent.

Goal:
Reach calorie target efficiently without overshooting.

Current state:
Calories consumed: {obs['calories_consumed']}
Target: {obs['calorie_target']}
Hunger: {obs['hunger']}

Available actions:
apple (95 cal), rice (200 cal), chicken (250 cal), snack (300 cal), skip (0 cal)

Respond with ONLY one word:
apple OR rice OR chicken OR snack OR skip
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

        if action not in ["apple", "rice", "chicken", "snack", "skip"]:
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

    try:
        res = requests.post(f"{ENV_URL}/reset").json()
        obs = res["observation"]

        for step in range(1, MAX_STEPS + 1):
            action = get_action_from_model(obs)

            res = requests.post(
                f"{ENV_URL}/step",
                json={"food": action}
            ).json()

            obs = res.get("observation", {})
            reward = res.get("reward", 0.0)
            done = res.get("done", False)
            error = res.get("info", {}).get("error")

            rewards.append(reward)
            steps_taken = step

            log_step(step, action, reward, done, error)

            if done:
                break

        if rewards:
            score = sum(rewards) / len(rewards)

        score = max(0.0, min(score, 1.0))
        success = score >= SUCCESS_SCORE_THRESHOLD

    finally:
        log_end(success, steps_taken, score, rewards)


if __name__ == "__main__":
    main()