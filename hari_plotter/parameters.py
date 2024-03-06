
from abc import ABC, abstractclassmethod
from typing import (Any, Dict, Iterator, List, Optional, Sequence, Tuple, Type,
                    Union)


class Parameter:
    @abstractclassmethod
    def validate(self) -> bool:
        pass


class ListParameter(Parameter):
    def __init__(self, name: str, parameter_name: str, arguments: List[str], comment: str = '') -> None:
        super().__init__()
        self.name = name
        self.parameter_name = parameter_name
        self.arguments = arguments
        self.comment = comment

    def validate(self, value: str) -> bool:
        return isinstance(value, str)


class BoolParameter(Parameter):
    def __init__(self, name: str, parameter_name: str, default_value: False, comment: str = '') -> None:
        super().__init__()
        self.name = name
        self.parameter_name = parameter_name
        self.default_value = default_value
        self.comment = comment

    def validate(self, value: bool) -> bool:
        return isinstance(value, bool)


class FloatParameter(Parameter):
    def __init__(self, name: str, parameter_name: str, default_value: float = 0.0, limits: Tuple[float] = (None, None),  comment: str = '') -> None:
        super().__init__()
        self.name = name
        self.parameter_name = parameter_name
        self.default_value = default_value
        self.limits = limits
        self.comment = comment

    def validate(self, value: float) -> bool:
        return isinstance(value, float) and (self.limits[0] is None or self.limits[0] <= value) and (self.limits[1] is None or self.limits[1] >= value)


class NoneOrFloatParameter(Parameter):
    def __init__(self, name: str, parameter_name: str, default_value: Union[float, None] = None, limits: Tuple[float] = (None, None),  comment: str = '') -> None:
        super().__init__()
        self.name = name
        self.parameter_name = parameter_name
        self.default_value = default_value
        self.limits = limits
        self.comment = comment

    def validate(self, value: Union[float, None]) -> bool:
        return value is None or (isinstance(value, float) and (self.limits[0] is None or self.limits[0] <= value) and (self.limits[1] is None or self.limits[1] >= value))


class NoneRangeParameter(Parameter):
    def __init__(self, name: str, parameter_name: str, default_min_value: Union[float, None] = None, default_max_value: Union[float, None] = None, limits: Tuple[float] = (None, None), comment: str = ''):
        super().__init__()
        self.name = name
        self.parameter_name = parameter_name
        self.default_min_value = default_min_value
        self.default_max_value = default_max_value
        self.limits = limits
        self.comment = comment

    def validate(self, value: Tuple[float, None]) -> bool:
        min_value, max_value = value
        # Check if both min and max values are None, floats, or one is None and the other is float
        if not ((min_value is None or isinstance(min_value, float)) and (max_value is None or isinstance(max_value, float))):
            return False

        # If both min and max values are not None, check that max is greater than min
        if min_value is not None and max_value is not None and max_value <= min_value:
            return False

        # Check that min and max values fall within the specified limits, if they are not None
        if min_value is not None and (self.limits[0] is not None and min_value < self.limits[0]):
            return False
        if max_value is not None and (self.limits[1] is not None and max_value > self.limits[1]):
            return False

        return True
