from typing import Union, Any
import numpy as np
import pandas as pd
import datetime as dt

DateType = dt.date
Int = Union[int, np.integer]
Float = Union[float, np.float32]
IntFloat = Union[Int, Float]

Array = np.ndarray  # ready to be used for n-dim data
Array1d = np.ndarray
Array2d = np.ndarray

FloatOrNpArray = Union[Float, np.ndarray]
Frame = pd.DataFrame
Series = pd.Series

BoolType = bool
Any = Union[Any]
