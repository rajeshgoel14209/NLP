https://pub.towardsai.net/llama-ocr-multimodal-rag-local-llm-python-project-easy-ai-chat-for-your-docs-058c0dd3b59f
https://towardsdatascience.com/improve-your-rag-context-recall-by-40-with-an-adapted-embedding-model-5d4a8f583f32
https://towardsdatascience.com/how-to-use-re-ranking-for-better-llm-rag-retrieval-243f89414266
https://towardsdatascience.com/multimodal-rag-process-any-file-type-with-ai-e6921342c903



pip install apscheduler
from fastapi import FastAPI
from apscheduler.schedulers.background import BackgroundScheduler
from datetime import datetime

app = FastAPI()

# Define the task to run weekly
def weekly_task():
    print(f"Weekly task triggered at {datetime.now()}")

# Initialize and configure the scheduler
scheduler = BackgroundScheduler()

# Add the weekly task
scheduler.add_job(weekly_task, 'interval', weeks=1)

# Start the scheduler when the application starts
@app.on_event("startup")
def start_scheduler():
    scheduler.start()
    print("Scheduler started")

# Shutdown the scheduler gracefully when the application stops
@app.on_event("shutdown")
def shutdown_scheduler():
    scheduler.shutdown()
    print("Scheduler stopped")

@app.get("/")
def read_root():
    return {"message": "FastAPI service with weekly task scheduler is running"}



scheduler.add_job(weekly_task, 'cron', day_of_week='mon', hour=9, minute=0)
## This schedules the task to run every Monday at 9:00 AM.

## Testing
## To test the weekly trigger during development, reduce the interval to seconds or minutes for faster feedback:

## python

## scheduler.add_job(weekly_task, 'interval', seconds=10)
