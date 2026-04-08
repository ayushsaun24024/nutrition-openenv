def clamp(x, low=0.0, high=1.0):
    return max(low, min(x, high))


def easy_grader(calories_consumed, calorie_target, steps_taken):
    diff = abs(calorie_target - calories_consumed)
    return max(0.0, min(1.0, 1.0 - (diff / calorie_target)))


def medium_grader(calories_consumed, calorie_target, steps_taken):
    diff = abs(calorie_target - calories_consumed)
    base = 1.0 - (diff / calorie_target)
    penalty = steps_taken / 10
    return max(0.0, min(1.0, base * (1 - 0.3 * penalty)))


def hard_grader(calories_consumed, calorie_target, steps_taken):
    diff = abs(calorie_target - calories_consumed)
    base = 1.0 - (diff / calorie_target)

    overshoot = 0.0
    if calories_consumed > calorie_target:
        overshoot = (calories_consumed - calorie_target) / calorie_target

    penalty = steps_taken / 10
    score = base - (0.5 * overshoot) - (0.3 * penalty)

    return max(0.0, min(1.0, score))


TASKS = {
    "easy": easy_grader,
    "medium": medium_grader,
    "hard": hard_grader,
}
