import uvicorn
from fastapi import FastAPI
from my_env.tasks import TASKS
from my_env.models import Action
from my_env.env import NutritionEnv
from pydantic import BaseModel, Field
from typing import List, Literal, Optional

app = FastAPI(title="Nutrition Environment API")

env = NutritionEnv()
current_task_name = "easy"

class StepRequest(BaseModel):
    food: Literal["apple", "rice", "chicken", "snack", "milk", "egg", "skip"] = Field(
        ...,
        description="Select a food item",
        example="rice"
    )

class ResetRequest(BaseModel):
    task: Optional[str] = "easy"

class RolloutRequest(BaseModel):
    actions: List[Literal["apple", "rice", "chicken", "snack", "milk", "egg", "skip"]]


@app.get("/tasks")
def list_tasks():
    return {
        "tasks": [
            {"id": t.name, "description": t.description}
            for t in TASKS.values()
        ]
    }


@app.post("/reset")
def reset(req: ResetRequest = None):
    global current_task_name
    current_task_name = (req.task if req and req.task in TASKS else "easy")
    obs = env.reset()
    return {
        "observation": obs.dict(),
        "reward": 0.0,
        "done": False,
        "info": {"task": current_task_name}
    }


@app.post("/step")
def step(req: StepRequest):
    obs, reward, done, info = env.step(Action(food=req.food))
    if done:
        task = TASKS[current_task_name]
        info["grader_score"] = task.grade(
            calories_consumed=obs.calories_consumed,
            calorie_target=obs.calorie_target,
            steps_taken=obs.step
        )
        info["task"] = current_task_name
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
