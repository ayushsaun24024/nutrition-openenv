def clamp(x, low=0.0, high=1.0):
    return max(low, min(x, high))

def grade_easy(calories_consumed, target):
    diff = abs(target - calories_consumed)
    score = 1.0 - (diff / target)
    return clamp(score)

def grade_medium(calories_consumed, target, steps_taken, max_steps):
    diff = abs(target - calories_consumed)
    base_score = 1.0 - (diff / target)

    efficiency_penalty = steps_taken / max_steps
    score = base_score * (1.0 - 0.3 * efficiency_penalty)

    return clamp(score)

def grade_hard(calories_consumed, target, steps_taken, max_steps):
    diff = abs(target - calories_consumed)
    base_score = 1.0 - (diff / target)

    overshoot_penalty = 0.0
    if calories_consumed > target:
        overshoot_penalty = (calories_consumed - target) / target

    efficiency_penalty = steps_taken / max_steps

    score = base_score - (0.5 * overshoot_penalty) - (0.3 * efficiency_penalty)

    return clamp(score)