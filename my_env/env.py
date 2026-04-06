from my_env.models import Observation, Action


FOOD_DB = {
    "apple": {"cal": 95, "protein": 0, "carbs": 25, "fat": 0, "sugar": 19},
    "rice": {"cal": 200, "protein": 4, "carbs": 45, "fat": 1, "sugar": 0},
    "chicken": {"cal": 250, "protein": 30, "carbs": 0, "fat": 5, "sugar": 0},
    "snack": {"cal": 300, "protein": 5, "carbs": 30, "fat": 15, "sugar": 20},
    "milk": {"cal": 120, "protein": 8, "carbs": 12, "fat": 5, "sugar": 10},
    "egg": {"cal": 80, "protein": 6, "carbs": 1, "fat": 5, "sugar": 0},
    "skip": {"cal": 0, "protein": 0, "carbs": 0, "fat": 0, "sugar": 0},
}


class NutritionEnv:
    def __init__(self, target_calories=2000, max_steps=10):
        self.target_calories = target_calories
        self.max_steps = max_steps
        self.reset()

    def reset(self):
        self.calories = 0.0
        self.protein = 0.0
        self.carbs = 0.0
        self.fat = 0.0
        self.sugar = 0.0

        self.step_count = 0
        self.done = False

        self.last_action = None
        self.repetition_count = 0

        return self._get_obs()

    def _get_obs(self):
        base_hunger = max(0.0, self.target_calories - self.calories) / self.target_calories
        satiety_effect = min(self.protein / 100.0, 0.3)
        hunger = max(0.0, base_hunger - satiety_effect)

        return Observation(
            calories_consumed=self.calories,
            calorie_target=self.target_calories,
            hunger=hunger,
            step=self.step_count,
        )

    def step(self, action: Action):
        if self.done:
            raise Exception("Episode finished")

        self.step_count += 1

        if action.food not in FOOD_DB:
            self.done = True
            return self._get_obs(), 0.0, True, {"error": "invalid_food"}

        food = FOOD_DB[action.food]

        self.calories += food["cal"]
        self.protein += food["protein"]
        self.carbs += food["carbs"]
        self.fat += food["fat"]
        self.sugar += food["sugar"]

        if action.food == self.last_action:
            self.repetition_count += 1
        else:
            self.repetition_count = 0
        self.last_action = action.food

        diff = abs(self.target_calories - self.calories)
        reward = 1.0 - (diff / self.target_calories)

        lower = 0.9 * self.target_calories
        upper = 1.1 * self.target_calories
        if lower <= self.calories <= upper:
            reward += 0.1

        if self.calories > self.target_calories:
            reward -= 0.5

        if self.step_count >= self.max_steps - 2 and self.calories < 0.7 * self.target_calories:
            reward -= 0.3

        if 50 <= self.protein <= 80:
            reward += 0.1
        elif self.protein > 80:
            reward -= 0.2

        if 50 <= self.protein <= 80 and self.carbs > 100:
            reward += 0.05

        if self.sugar > 60:
            reward -= 0.15

        if self.fat > 70:
            reward -= 0.1

        if self.repetition_count == 0:
            reward += 0.1

        if self.repetition_count >= 2:
            reward -= 0.3

        if self.step_count <= 3 and food["cal"] > 250:
            reward -= 0.05

        if self.step_count >= 8 and abs(self.target_calories - self.calories) > 200:
            reward -= 0.1

        reward = max(0.0, min(reward, 1.0))

        if self.step_count >= self.max_steps:
            self.done = True

        if abs(self.target_calories - self.calories) < 50:
            self.done = True

        return self._get_obs(), reward, self.done, {}

    def state(self):
        return {
            "calories": self.calories,
            "protein": self.protein,
            "carbs": self.carbs,
            "fat": self.fat,
            "sugar": self.sugar,
            "step": self.step_count,
        }