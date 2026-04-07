def clamp(x, low=0.0, high=1.0):
    return max(low, min(x, high))


class EasyTask:
    name = "easy"
    description = "Reach the daily calorie target. Focus: basic correctness."
    max_steps = 10

    def grade(self, calories_consumed: float, calorie_target: float, steps_taken: int) -> float:
        diff = abs(calorie_target - calories_consumed)
        score = 1.0 - (diff / calorie_target)
        return clamp(score)


class MediumTask:
    name = "medium"
    description = "Reach target efficiently. Penalizes unnecessary steps."
    max_steps = 10

    def grade(self, calories_consumed: float, calorie_target: float, steps_taken: int) -> float:
        diff = abs(calorie_target - calories_consumed)
        base_score = 1.0 - (diff / calorie_target)
        efficiency_penalty = steps_taken / self.max_steps
        score = base_score * (1.0 - 0.3 * efficiency_penalty)
        return clamp(score)


class HardTask:
    name = "hard"
    description = "Precise calorie matching. Strong penalty for overshooting."
    max_steps = 10

    def grade(self, calories_consumed: float, calorie_target: float, steps_taken: int) -> float:
        diff = abs(calorie_target - calories_consumed)
        base_score = 1.0 - (diff / calorie_target)
        overshoot_penalty = 0.0
        if calories_consumed > calorie_target:
            overshoot_penalty = (calories_consumed - calorie_target) / calorie_target
        efficiency_penalty = steps_taken / self.max_steps
        score = base_score - (0.5 * overshoot_penalty) - (0.3 * efficiency_penalty)
        return clamp(score)


TASKS = {
    "easy":   EasyTask(),
    "medium": MediumTask(),
    "hard":   HardTask(),
}
