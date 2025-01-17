from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import os

# Initialize FastAPI app
app = FastAPI()

# Define the input data model
class Feedback(BaseModel):
    query: str
    context: dict  # Context with metadata
    llm_response: str
    user_feedback: str

# Path to store the Excel file
FEEDBACK_FILE = "feedback_data.xlsx"

# Function to initialize the Excel file if not exists
def initialize_excel_file(file_path):
    if not os.path.exists(file_path):
        # Create a new DataFrame with column names
        df = pd.DataFrame(columns=["Query", "Context", "LLM Response", "User Feedback"])
        df.to_excel(file_path, index=False)

# Initialize the feedback file
initialize_excel_file(FEEDBACK_FILE)

@app.post("/submit-feedback/")
async def submit_feedback(feedback: Feedback):
    """
    Endpoint to accept feedback and store it in an Excel file.

    Args:
        feedback (Feedback): Input query, context, LLM response, and user feedback.

    Returns:
        dict: Success message with the added data.
    """
    try:
        # Load the existing Excel file
        df = pd.read_excel(FEEDBACK_FILE)

        # Append the new feedback as a row
        new_row = {
            "Query": feedback.query,
            "Context": feedback.context,
            "LLM Response": feedback.llm_response,
            "User Feedback": feedback.user_feedback,
        }
        df = df.append(new_row, ignore_index=True)

        # Save the updated DataFrame back to the Excel file
        df.to_excel(FEEDBACK_FILE, index=False)

        return {"message": "Feedback submitted successfully!", "data": new_row}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")


@app.get("/feedback/")
async def get_feedback():
    """
    Endpoint to fetch all feedback from the Excel file.

    Returns:
        dict: List of all feedback data.
    """
    try:
        # Load the Excel file
        df = pd.read_excel(FEEDBACK_FILE)

        # Convert the DataFrame to a list of dictionaries
        feedback_data = df.to_dict(orient="records")

        return {"feedback": feedback_data}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")


        
