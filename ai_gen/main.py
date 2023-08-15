import logging
import time

import torch
from fastapi import BackgroundTasks, FastAPI

from ai_gen.generate_video import GenerateVideoArgs, generate_and_send_video_task

logging.warning("text")

app = FastAPI()


@app.get("/health")
async def health():
    return {"status": "OK"}


@app.post("/generate_video")
async def generate_video(args: GenerateVideoArgs, background_tasks: BackgroundTasks):
    if torch.cuda.is_available() and not args.cpu:
        args.device = "cuda"
    background_tasks.add_task(generate_and_send_video_task, args, file_name=time.strftime("%Y_%m_%d_%H.%M.%S"))
    return {"status": f"started generating video, recieved args: {args.model_dump()}"}

