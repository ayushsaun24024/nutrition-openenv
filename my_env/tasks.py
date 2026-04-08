def clamp(score):
    return max(0.01, min(0.99, float(score)))


def easy_grader(observation, info=None):
    calories_consumed = observation.get("calories_consumed", 0)
    calorie_target = observation.get("calorie_target", 1)

    diff = abs(calorie_target - calories_consumed)
    score = 1.0 - (diff / calorie_target)

    return clamp(score)


def medium_grader(observation, info=None):
    calories_consumed = observation.get("calories_consumed", 0)
    calorie_target = observation.get("calorie_target", 1)
    steps_taken = observation.get("step", 10)

    diff = abs(calorie_target - calories_consumed)
    base = 1.0 - (diff / calorie_target)
    penalty = steps_taken / 10

    score = base * (1 - 0.3 * penalty)

    return clamp(score)


def hard_grader(observation, info=None):
    calories_consumed = observation.get("calories_consumed", 0)
    calorie_target = observation.get("calorie_target", 1)
    steps_taken = observation.get("step", 10)

    diff = abs(calorie_target - calories_consumed)
    base = 1.0 - (diff / calorie_target)

    overshoot = 0.0
    if calories_consumed > calorie_target:
        overshoot = (calories_consumed - calorie_target) / calorie_target

    penalty = steps_taken / 10

    score = base - (0.5 * overshoot) - (0.3 * penalty)

    return clamp(score)


TASKS = {
    "easy": easy_grader,
    "medium": medium_grader,
    "hard": hard_grader,
}
