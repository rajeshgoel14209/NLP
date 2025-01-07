I need the output in the form of a list of dictionaries. Each dictionary should have the following keys: ["title", "author", "year"]. Ensure the values match the requested format.

Here is an example of the desired output format:
[
    {"title": "Book Title 1", "author": "Author Name 1", "year": 2000},
    {"title": "Book Title 2", "author": "Author Name 2", "year": 2005}
]

Now, based on the following input, generate the corresponding list of dictionaries:

Input:
1. "To Kill a Mockingbird" by Harper Lee, published in 1960.
2. "1984" by George Orwell, published in 1949.
3. "The Great Gatsby" by F. Scott Fitzgerald, published in 1925.

Output:


Expected LLM Response:

[    {"title": "To Kill a Mockingbird", "author": "Harper Lee", "year": 1960},    {"title": "1984", "author": "George Orwell", "year": 1949},    {"title": "The Great Gatsby", "author": "F. Scott Fitzgerald", "year": 1925}]


Tips for Prompting
Be Explicit: Clearly describe the required keys and values.
Provide an Example: Show the LLM the exact format you want.
Repeat Key Points: Reinforce the desired structure in the prompt.
Validate Format: Include instructions to ensure no extra text appears outside the list of dictionaries.
