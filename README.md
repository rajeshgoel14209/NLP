# Step 1: First filter - Select documents where 'category' is 'technology'
first_filter = {"category": {"$eq": "technology"}}
step1_results = collection.query(
    query_texts=["AI advancements"],
    n_results=10,
    where=first_filter
)

# Extract IDs of filtered documents
filtered_ids = step1_results.get("ids", [])

# Step 2: Apply second filter - Filter documents from Step 1 where 'year' is >= 2020
if filtered_ids:
    second_filter = {
        "year": {"$gte": 2020},
        "id": {"$in": filtered_ids}  # Only apply to previously filtered results
    }
    
    step2_results = collection.query(
        query_texts=["AI advancements"],
        n_results=5,
        where=second_filter
    )

    print(step2_results)
