from pydantic import BaseModel

class Observation(BaseModel):
    calories_consumed: float
    calorie_target: float
    hunger: float
    step: int

class Action(BaseModel):
    food: str

class Reward(BaseModel):
    value: float