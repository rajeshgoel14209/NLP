def filter_dicts(list_of_dicts, filter_criteria):
    """
    Filters a list of dictionaries based on a given dictionary containing subset keys and values.

    :param list_of_dicts: List[Dict] - The list of dictionaries to filter.
    :param filter_criteria: Dict - A dictionary containing the keys and values to filter on.
    :return: List[Dict] - A list of dictionaries that match the filter criteria.
    """
    return [
        d for d in list_of_dicts 
        if all(d.get(key) == value for key, value in filter_criteria.items())
    ]
