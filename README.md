from apscheduler.schedulers.background import BackgroundScheduler
import os
import signal
import time

def restart_fastapi():
    print("Restarting FastAPI...")
    os.kill(os.getpid(), signal.SIGTERM)  # Kills and restarts FastAPI

scheduler = BackgroundScheduler()
scheduler.add_job(restart_fastapi, "interval", hours=6)  # Restart every 6 hours
scheduler.start()

# FastAPI App
from fastapi import FastAPI
app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "FastAPI is running"}

import uvicorn
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
