from my_env.models import Observation, Action


FOOD_DB = {
    "apple": 95,
    "rice": 200,
    "chicken": 250,
    "snack": 300,
    "skip": 0,
}


class NutritionEnv:
    def __init__(self, target_calories=2000, max_steps=10):
        self.target_calories = target_calories
        self.max_steps = max_steps
        self.reset()

    def reset(self):
        self.calories_consumed = 0.0
        self.step_count = 0
        self.done = False
        return self._get_observation()

    def _get_observation(self):
        hunger = max(0.0, self.target_calories - self.calories_consumed) / self.target_calories

        return Observation(
            calories_consumed=self.calories_consumed,
            calorie_target=self.target_calories,
            hunger=hunger,
            step=self.step_count,
        )

    def step(self, action: Action):
        if self.done:
            raise Exception("Episode already finished. Call reset().")

        self.step_count += 1

        if action.food not in FOOD_DB:
            reward = -1.0
            self.done = True
            return self._get_observation(), reward, self.done, {"error": "invalid_food"}

        calories = FOOD_DB[action.food]
        self.calories_consumed += calories

        diff = abs(self.target_calories - self.calories_consumed)
        reward = 1.0 - (diff / self.target_calories)

        if self.calories_consumed > self.target_calories:
            reward -= 0.5

        reward = max(0.0, min(reward, 1.0))

        if self.step_count >= self.max_steps:
            self.done = True

        if abs(self.target_calories - self.calories_consumed) < 50:
            self.done = True

        return self._get_observation(), reward, self.done, {}

    def state(self):
        return {
            "calories_consumed": self.calories_consumed,
            "step_count": self.step_count,
            "done": self.done,
        }
