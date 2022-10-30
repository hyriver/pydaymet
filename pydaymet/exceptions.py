"""Customized PyDaymet exceptions."""
from __future__ import annotations

from typing import Any, Generator, Iterable


class InputValueError(Exception):
    """Exception raised for invalid input.

    Parameters
    ----------
    inp : str
        Name of the input parameter
    valid_inputs : tuple
        List of valid inputs
    """

    def __init__(self, inp: str, valid_inputs: Iterable[Any] | Generator[str, None, None]) -> None:
        valid = ", ".join(str(i) for i in valid_inputs)
        self.message = f"Valid values for {inp} are:\n{valid}"
        super().__init__(self.message)

    def __str__(self) -> str:
        return self.message


class InputTypeError(Exception):
    """Exception raised when a function argument type is invalid.

    Parameters
    ----------
    arg : str
        Name of the function argument
    valid_type : str
        The valid type of the argument
    example : str, optional
        An example of a valid form of the argument, defaults to None.
    """

    def __init__(self, arg: str, valid_type: str, example: str | None = None) -> None:
        self.message = f"The {arg} argument should be of type {valid_type}"
        if example is not None:
            self.message += f":\n{example}"
        super().__init__(self.message)

    def __str__(self) -> str:
        return self.message


class InputRangeError(Exception):
    """Exception raised when a function argument is not in the valid range.

    Parameters
    ----------
    variable : str
        Variable with invalid value
    valid_range : str
        Valid range
    """

    def __init__(self, variable: str, valid_range: str) -> None:
        self.message = f"Valid range for {variable} is {valid_range}."
        super().__init__(self.message)

    def __str__(self) -> str:
        return self.message


class MissingItemError(Exception):
    """Exception raised when a required item is missing.

    Parameters
    ----------
    missing : list
        A list of missing items.
    """

    def __init__(self, missing: list[str]) -> None:
        self.message = "The following items are missing:\n" + f"{', '.join(missing)}"
        super().__init__(self.message)

    def __str__(self) -> str:
        return self.message


class MissingCRSError(Exception):
    """Exception raised when input GeoDataFrame is missing CRS."""

    def __init__(self) -> None:
        self.message = "The input GeoDataFrame is missing CRS."
        super().__init__(self.message)

    def __str__(self) -> str:
        return self.message
