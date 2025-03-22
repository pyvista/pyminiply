import numpy as np
from numpy.typing import NDArray

def load_ply(
    filename: str,
    read_normals: bool = True,
    read_uv: bool = True,
    read_color: bool = True,
) -> tuple[
    NDArray[np.float32],
    NDArray[np.int32],
    NDArray[np.float32],
    NDArray[np.float32],
    NDArray[np.uint8],
]: ...
