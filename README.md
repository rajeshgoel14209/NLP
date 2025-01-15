import re

def extract_case_ids(query):
    """
    Extract all case IDs from a user query.
    
    Args:
        query (str): The user query string.
    
    Returns:
        list: A list of all matched case IDs.
    """
    # Define a regex pattern to match various formats of case IDs
    # Example: CASE-12345, Case_54321, CASE123, etc.
    case_id_pattern = r'\bCASE[-_]?\d+\b'

    # Use re.findall to extract all matches
    case_ids = re.findall(case_id_pattern, query, flags=re.IGNORECASE)

    # Normalize case IDs (e.g., convert to uppercase)
    case_ids = [case_id.upper() for case_id in case_ids]
    
    return case_ids



    # Example queries
queries = [
    "Please check CASE-12345 and Case_67890.",
    "The IDs are CASE123, case_456, and CASE-789.",
    "No case IDs here.",
    "Invalid examples: USER-123, CASE-."
]

# Test the function
for query in queries:
    print(f"Query: {query}")
    print(f"Extracted Case IDs: {extract_case_ids(query)}\n")

    
