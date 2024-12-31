def evaluate_condition(item: Dict[str, Any], condition: Dict[str, Any]) -> bool:
    """
    Evaluates a single condition on an item.

    Args:
        item (Dict[str, Any]): The item to evaluate.
        condition (Dict[str, Any]): A condition with 'field', 'operator', and 'value'.

    Returns:
        bool: Whether the condition is satisfied.
    """
    field = condition["field"]
    operator = condition["operator"]
    value = condition["value"]

    if field not in item:
        return False

    if operator == "==":
        return item[field] == value
    elif operator == ">":
        return item[field] > value
    elif operator == "<":
        return item[field] < value
    elif operator == ">=":
        return item[field] >= value
    elif operator == "<=":
        return item[field] <= value
    elif operator == "!=":
        return item[field] != value
    else:
        raise ValueError(f"Unsupported operator: {operator}")

def evaluate_all_conditions(item: Dict[str, Any], conditions: List[Dict[str, Any]]) -> bool:
    """
    Evaluates a list of conditions on an item. Combines the conditions with `and`.

    Args:
        item (Dict[str, Any]): The item to evaluate.
        conditions (List[Dict[str, Any]]): A list of conditions.

    Returns:
        bool: Whether all conditions are satisfied.
    """
    return all(evaluate_condition(item, condition) for condition in conditions)
