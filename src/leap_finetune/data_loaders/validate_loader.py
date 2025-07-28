from functools import wraps
from typing import Any
from datasets import Dataset


def validate_data_loader(func):
    """Decorator that validates function returns tuple[Dataset, Dataset] for custom data loaders"""

    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> tuple[Dataset, Dataset]:
        result = func(*args, **kwargs)

        # Validate structure
        if not isinstance(result, tuple) or len(result) != 2:
            raise TypeError(
                f"{func.__name__} must return tuple[Dataset, Dataset], got {type(result)}"
            )

        train, test = result
        if not isinstance(train, Dataset) or not isinstance(test, Dataset):
            raise TypeError(f"{func.__name__} must return tuple of Dataset instances")

        return result

    return wrapper
