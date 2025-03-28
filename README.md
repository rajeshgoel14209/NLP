SyntaxError
    Raised when there is a syntax mistake in the code.
	
TypeError
    Raised when an operation is performed on an inappropriate type.
	
ValueError 
    Raised when an argument has the right type but an invalid value.

IndexError
    Raised when accessing an index that is out of range.

KeyError
    Raised when accessing a dictionary key that does not exist.

AttributeError
    Raised when an invalid attribute is accessed on an object.
	
NameError
    Raised when trying to access an undefined variable.
	
FileNotFoundError
    Raised when trying to open a non-existing file.
	
PermissionError
    Raised when trying to access a file without proper permissions.

OverflowError
     Raised when a calculation exceeds the maximum limit.

ImportError
    Raised when a module cannot be imported.
	
ModuleNotFoundError 
    A subclass of ImportError when a module is not found.

example - 
	
try:
    print(10 / 0)
except ZeroDivisionError as e:
    print("Error:", e)

@app.errorhandler(Exception)
def handle_exception(e):
    return jsonify({"error": str(e)}), 500  # Returns HTTP 500 instead of crashing

class ConfigError(Exception):
    """Raised when the config file is invalid or missing"""

try:
    raise ConfigError("Invalid config format")
except ConfigError as e:
    print(f"Configuration Error: {e}")

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
import os

app = FastAPI()

# Define the directory where logs are stored
LOG_DIR = "logs"

@app.get("/download-log/{filename}")
async def download_log(filename: str):
    """API to download a log file from the server."""
    file_path = os.path.join(LOG_DIR, filename)

    # Validate the file path (prevent directory traversal)
    if not os.path.abspath(file_path).startswith(os.path.abspath(LOG_DIR)):
        raise HTTPException(status_code=403, detail="Access denied!")

    # Check if the file exists
    if not os.path.isfile(file_path):
        raise HTTPException(status_code=404, detail="Log file not found")

    # Serve the log file as a download
    return FileResponse(file_path, media_type="text/plain", filename=filename)


    http://127.0.0.1:8000/download-log/app.log
