from _typing import Any, Int
import numpy as np


####### Checks #######
def is_np_array(arg: Any) -> bool:
    """Check whether the argument is `np.ndarray`."""
    return isinstance(arg, np.ndarray)

# ######## Asset ######### #
def assert_not_none(arg: Any) -> None:
    """Raise exception if the argument is None."""
    if arg is None:
        raise AssertionError(f"Argument cannot be None")


def assert_ndim(arg: Any, ndims: Int) -> None:
    """Raise exception if the argument has a different number of dimensions than `ndims`."""
    if np.ndim(arg) != ndims:
        raise AssertionError(f"Number of dimensions must be {ndims}, not {np.ndim(arg)}")

