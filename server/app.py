import uvicorn
from fastapi import FastAPI
from my_env.models import Action
from typing import List, Literal
from my_env.env import NutritionEnv
from pydantic import BaseModel, Field

app = FastAPI(title="Nutrition Environment API")

env = NutritionEnv()

class StepRequest(BaseModel):
    food: Literal["apple", "rice", "chicken", "snack", "milk", "egg", "skip"] = Field(
        ...,
        description="Select a food item",
        example="rice"
    )
    
class RolloutRequest(BaseModel):
    actions: List[Literal["apple", "rice", "chicken", "snack", "milk", "egg", "skip"]]


@app.post("/reset")
def reset():
    obs = env.reset()
    return {
        "observation": obs.dict(),
        "reward": 0.0,
        "done": False,
        "info": {}
    }


@app.post("/step")
def step(req: StepRequest):
    obs, reward, done, info = env.step(Action(food=req.food))
    return {
        "observation": obs.dict(),
        "reward": reward,
        "done": done,
        "info": info
    }


@app.get("/state")
def state():
    return env.state()

@app.post("/rollout")
def rollout(req: RolloutRequest):
    obs = env.reset()
    trajectory = []

    for step_idx, action in enumerate(req.actions, start=1):
        obs, reward, done, info = env.step(Action(food=action))

        trajectory.append({
            "step": step_idx,
            "action": action,
            "observation": obs.dict(),
            "reward": reward,
            "done": done
        })

        if done:
            break

    return {
        "trajectory": trajectory,
        "final_state": env.state()
    }

@app.get("/")
def root():
    return {"message": "NutritionEnv API is running"}

def main():
    uvicorn.run("server.app:app", host="0.0.0.0", port=8000)

if __name__ == "__main__":
    main()