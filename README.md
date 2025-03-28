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
