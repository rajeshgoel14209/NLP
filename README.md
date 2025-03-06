from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel
from typing import Optional

app = FastAPI()

# Define request body model
class ChatRequest(BaseModel):
    query: str
    chatInitiator: Optional[str] = None

# Define the POST API
@app.post("/jarvis/chat/query")
async def jarvis_chat(
    chat_request: ChatRequest,
    caseID: str = Header(None),
    userID: str = Header(None),
    username: str = Header(None)
):
    # Validate required headers
    if not caseID or not userID or not username:
        raise HTTPException(status_code=400, detail="Missing required headers: caseID, userID, username")

    return {
        "message": "Chat query received successfully!",
        "headers": {
            "caseID": caseID,
            "userID": userID,
            "username": username
        },
        "payload": chat_request
    }
