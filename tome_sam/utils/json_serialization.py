def convert_to_serializable_dict(obj):
    if obj is None:
        return None

    # Handle basic types directly
    if isinstance(obj, (str, int, float, bool)):
        return obj

    # Handle lists and tuples
    if isinstance(obj, (list, tuple)):
        return [convert_to_serializable_dict(item) for item in obj]

    # Handle dictionaries
    if isinstance(obj, dict):
        return {k: convert_to_serializable_dict(v) for k, v in obj.items()}

    # Handle objects with __dict__ attribute
    if hasattr(obj, '__dict__'):
        return {k: convert_to_serializable_dict(v) for k, v in vars(obj).items()}

    # Fallback to string representation for other types
    return str(obj)