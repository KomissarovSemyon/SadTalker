from fastapi import FastAPI, BackgroundTasks
from ai_gen.generate_video import generate_video_task

app = FastAPI()


@app.get("/health")
async def health():
    return {"status": "OK"}


@app.post("/generate_video")
async def generate_video(background_tasks: BackgroundTasks):
    background_tasks.add_task(generate_video_task)
    return {"status": "started generating video"}

