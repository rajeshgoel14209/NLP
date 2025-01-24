def filter_dict_indices(list_of_dicts, filter_criteria):
    """
    Filters a list of dictionaries based on a given dictionary containing subset keys and values,
    and returns the indices of matching dictionaries.

    :param list_of_dicts: List[Dict] - The list of dictionaries to filter.
    :param filter_criteria: Dict - A dictionary containing the keys and values to filter on.
    :return: List[int] - A list of indices for dictionaries that match the filter criteria.
    """
    return [
        i for i, d in enumerate(list_of_dicts)
        if all(d.get(key) == value for key, value in filter_criteria.items())
    ]

# Example Usage
data = [
    {"id": 1, "name": "Alice", "age": 30},
    {"id": 2, "name": "Bob", "age": 25},
    {"id": 3, "name": "Alice", "age": 35}
]
