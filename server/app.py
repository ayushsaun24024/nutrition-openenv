from fastapi import FastAPI
from pydantic import BaseModel
from my_env.env import NutritionEnv
from my_env.models import Action
import uvicorn

app = FastAPI()

env = NutritionEnv()


class StepRequest(BaseModel):
    food: str


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
    try:
        action = Action(food=req.food)
        obs, reward, done, info = env.step(action)

        return {
            "observation": obs.dict(),
            "reward": reward,
            "done": done,
            "info": info
        }

    except Exception as e:
        return {
            "observation": {},
            "reward": 0.0,
            "done": True,
            "info": {"error": str(e)}
        }


@app.get("/state")
def state():
    return env.state()

@app.get("/")
def root():
    return {"message": "NutritionEnv API is running"}

def main():
    uvicorn.run("server.app:app", host="0.0.0.0", port=8000)

if __name__ == "__main__":
    main()