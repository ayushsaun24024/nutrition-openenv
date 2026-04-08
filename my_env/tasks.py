def clamp(x, low=0.0, high=1.0):
    return max(low, min(x, high))


def easy_grader(observation, info=None):
    calories_consumed = observation.get("calories_consumed", 0)
    calorie_target = observation.get("calorie_target", 1)

    diff = abs(calorie_target - calories_consumed)
    return max(0.0, min(1.0, 1.0 - (diff / calorie_target)))


def medium_grader(observation, info=None):
    calories_consumed = observation.get("calories_consumed", 0)
    calorie_target = observation.get("calorie_target", 1)
    steps_taken = observation.get("step", 10)

    diff = abs(calorie_target - calories_consumed)
    base = 1.0 - (diff / calorie_target)
    penalty = steps_taken / 10

    return max(0.0, min(1.0, base * (1 - 0.3 * penalty)))


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

    return max(0.0, min(1.0, score))


TASKS = {
    "easy": easy_grader,
    "medium": medium_grader,
    "hard": hard_grader,
}
