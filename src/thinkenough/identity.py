"""Stable question IDs: preserve integer IDs and opaque string IDs."""


def valid_question_id(value):
    return (type(value) is int and value >= 0) or (
        isinstance(value, str) and bool(value.strip()) and "/" not in value
    )


def question_sort_key(value):
    # Keep integer and string namespaces distinct, including 1 versus "1".
    return (0, value) if type(value) is int else (1, str(value))
