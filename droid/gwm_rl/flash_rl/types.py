from typing import Any, Union

import numpy as np
import numpy.typing as npt
import torch

NDArray = npt.NDArray[Any]
F32NDArray = npt.NDArray[np.float32]
Tensor = Union[NDArray, torch.Tensor]
