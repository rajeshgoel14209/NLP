def extract_case_ids(query):
    """
    Extract all case IDs or business case IDs from a user query.
    
    Args:
        query (str): The user query string.
    
    Returns:
        list: A list of all matched case IDs or business case IDs.
    """
    # Define regex pattern for case IDs and business case IDs
    case_id_pattern = r'\b(?:business\s+case|case(?:\s?id)?)\s+(\d+)\b'
    
    # Use re.findall to extract all matches
    matches = re.findall(case_id_pattern, query, flags=re.IGNORECASE)
    
    return matches

    # Example queries
queries = [
    "What is the review date for case id 786868 and cag id 768667689678969?",
    "What is the review date for case 786868 and cag id 768667689678969?",
    "What is the review date for business case 786868 and cag id 768667689678969?",
    "No case IDs here, just CAG IDs like 768667689678969.",
]

# Test the function
for i, query in enumerate(queries, 1):
    print(f"Query {i}: {query}")
    print(f"Extracted Case IDs: {extract_case_ids(query)}\n")
