import time

from fastapi import BackgroundTasks, FastAPI

from ai_gen.generate_video import GenerateVideoArgs, generate_video_task

app = FastAPI()


@app.get("/health")
async def health():
    return {"status": "OK"}


@app.post("/generate_video")
async def generate_video(args: GenerateVideoArgs, background_tasks: BackgroundTasks):
    background_tasks.add_task(generate_video_task, args, file_name=time.strftime("%Y_%m_%d_%H.%M.%S"))
    return {"status": "started generating video"}

