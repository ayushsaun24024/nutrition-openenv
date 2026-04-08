import requests
import os
from openai import OpenAI

API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME   = os.getenv("MODEL_NAME", "gpt-4o-mini")
API_KEY      = os.getenv("HF_TOKEN") or os.getenv("OPENAI_API_KEY")

ENV_URL    = os.getenv("ENV_URL", "https://ayushsaun-nutrition-openenv.hf.space")
BENCHMARK  = "nutrition-env"

MAX_STEPS               = 10
SUCCESS_SCORE_THRESHOLD = 0.7

TASKS = ["easy", "medium", "hard"]

client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)


def safe_score(value):
    return max(1e-6, min(float(value), 1 - 1e-6))


def log_start(task):
    print(f"[START] task={task} env={BENCHMARK} model={MODEL_NAME}", flush=True)


def log_step(step, action, reward, done, error):
    error_val = error if error else "null"
    done_val  = str(done).lower()
    print(f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}", flush=True)


def log_end(success, steps, score, rewards):
    rewards_str = ",".join(f"{safe_score(r):.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}", flush=True)


def get_action_from_model(obs, prev_reward=None, reward_trend=None, action_history=None):
    if action_history is None:
        action_history = []

    calories_consumed  = obs.get('calories_consumed', 0)
    calorie_target     = obs.get('calorie_target', 2000)
    calories_remaining = calorie_target - calories_consumed
    step               = obs.get('step', 0)
    hunger             = obs.get('hunger', 0.5)

    last_action        = action_history[-1] if action_history else "none"
    last_3             = action_history[-3:] if action_history else []
    consecutive_repeat = all(a == last_action for a in last_3) and len(last_3) == 3

    history_str = " → ".join(action_history) if action_history else "none yet"

    blocked = set()
    if last_action != "none":
        blocked.add(last_action)
    if action_history.count("chicken") >= 2:
        blocked.add("chicken")
    if action_history.count("snack") >= 2:
        blocked.add("snack")
    if action_history.count("apple") >= 3:
        blocked.add("apple")

    allowed = [f for f in ["apple", "rice", "chicken", "snack", "milk", "egg", "skip"]
               if f not in blocked]

    prompt = f"""
You are a nutrition agent. Pick ONE food to eat right now.

=== FOOD TABLE ===
food     | cal | protein | carbs | fat | sugar
---------|-----|---------|-------|-----|------
apple    |  95 |    0g   |  25g  |  0g |  19g
rice     | 200 |    4g   |  45g  |  1g |   0g
chicken  | 250 |   30g   |   0g  |  5g |   0g
snack    | 300 |    5g   |  30g  | 15g |  20g
milk     | 120 |    8g   |  12g  |  5g |  10g
egg      |  80 |    6g   |   1g  |  5g |   0g
skip     |   0 |    0g   |   0g  |  0g |   0g

=== PENALTIES (AVOID) ===
- Protein > 80g total   → PENALTY
- Sugar   > 60g total   → PENALTY
- Fat     > 70g total   → PENALTY
- Overshoot calories    → BIG PENALTY
- Same food 3x in a row → PENALTY

=== CURRENT STATE ===
- Step:              {step} / {MAX_STEPS}  (steps left: {MAX_STEPS - step})
- Calories so far:   {calories_consumed:.0f} / {calorie_target:.0f}
- Calories left:     {calories_remaining:.0f}
- Hunger:            {hunger:.2f}
- Previous reward:   {prev_reward if prev_reward is not None else "N/A"}
- Reward trend:      {reward_trend if reward_trend else "N/A"}

=== YOUR MEAL HISTORY (what you already ate) ===
{history_str}

=== LAST ACTION: {last_action.upper()} — DO NOT PICK THIS AGAIN ===

=== BLOCKED (do not use these): {", ".join(blocked) if blocked else "none"} ===
=== ALLOWED choices only: {", ".join(allowed)} ===

=== ENDGAME RULES (step >= 8) ===
- If calories_remaining < 100 → pick egg or skip
- If calories_remaining 100–200 → pick milk or egg
- If calories_remaining > 200 → pick rice

Pick ONE word from ALLOWED list only: {", ".join(allowed)}
"""

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": "You are a nutrition agent. Reply with exactly ONE word from the allowed list."},
                {"role": "user",   "content": prompt},
            ],
            temperature=0.4,
            max_tokens=10,
        )
        action = response.choices[0].message.content.strip().lower()
        if action not in allowed:
            return allowed[0]
        return action
    except Exception as e:
        print(f"[DEBUG] OpenAI error: {e}", flush=True)
        return allowed[0] if allowed else "rice"


def run_task(task_name):
    rewards          = []
    steps_taken      = 0
    success          = False
    score            = 0.0
    prev_reward      = None
    prev_prev_reward = None
    action_history   = []

    log_start(task_name)

    try:
        res = requests.post(f"{ENV_URL}/reset", json={"task": task_name}, timeout=30).json()
        obs = res.get("observation") or {}

        for step in range(1, MAX_STEPS + 1):
            if prev_reward is None or prev_prev_reward is None:
                trend = "N/A"
            else:
                trend = (
                    "increasing" if prev_reward > prev_prev_reward else
                    "decreasing" if prev_reward < prev_prev_reward else
                    "stable"
                )

            action = get_action_from_model(
                obs,
                prev_reward=prev_reward,
                reward_trend=trend,
                action_history=action_history
            )

            res    = requests.post(f"{ENV_URL}/step", json={"food": action}, timeout=30).json()
            obs    = res.get("observation", {})
            reward = res.get("reward", 0.0)
            done   = res.get("done", False)
            error  = res.get("info", {}).get("error") or None

            rewards.append(reward)
            steps_taken = step
            action_history.append(action)

            log_step(step, action, reward, done, error)

            prev_prev_reward = prev_reward
            prev_reward      = reward

            if done:
                break

        score   = sum(rewards) / len(rewards) if rewards else 0.0
        score   = max(1e-6, min(score, 1 - 1e-6))
        success = score >= SUCCESS_SCORE_THRESHOLD

    except Exception as e:
        print(f"[DEBUG] Fatal error in task {task_name}: {e}", flush=True)

    finally:
        log_end(success, steps_taken, score, rewards)


def main():
    for task_name in TASKS:
        run_task(task_name)


if __name__ == "__main__":
    main()
