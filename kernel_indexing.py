import numpy as np
from dataclasses import dataclass
from icecream import ic


@dataclass
class GridDim:
    x: float = 1
    y: float = 1
    z: float = 1


@dataclass
class BlockIdx:
    x: float
    y: float
    z: float


@dataclass
class BlockDim:
    x: float = 1
    y: float = 1
    z: float = 1


@dataclass
class ThreadIdx:
    x: float
    y: float
    z: float


if __name__ == "__main__":
    A = np.arange(6 * 12).reshape(6, 12)
    ic(A)

    # Thread block
    blockDim = BlockDim(3, 4, 1)
    # Block grid
    # TODO: Use cdiv
    gridDim = GridDim(A.shape[0] // blockDim.x, A.shape[1] // blockDim.y, 1)

    B = np.zeros_like(A)
    C = np.zeros_like(A)
    D = np.zeros_like(A)
    E = np.zeros_like(A)

    def kernel(blockIdx: BlockIdx, threadIdx: ThreadIdx):
        i = blockIdx.x * blockDim.x + threadIdx.x
        j = blockIdx.y * blockDim.y + threadIdx.y
        B[i, j] = i + j * blockDim.x * gridDim.x
        C[i, j] = j + i * blockDim.y * gridDim.y

        blockNumInGrid = blockIdx.x + gridDim.x * blockIdx.y
        threadsPerBlock = blockDim.x * blockDim.y
        threadNumInBlock = threadIdx.x + blockDim.x * threadIdx.y
        D[i, j] = blockNumInGrid * threadsPerBlock + threadNumInBlock

        blockNumInGrid = blockIdx.y + gridDim.y * blockIdx.x
        threadNumInBlock = threadIdx.y + blockDim.y * threadIdx.x
        E[i, j] = blockNumInGrid * threadsPerBlock + threadNumInBlock

    for bk in range(gridDim.z):
        for bj in range(gridDim.y):
            for bi in range(gridDim.x):
                blockIdx = BlockIdx(bi, bj, bk)
                for tk in range(blockDim.z):
                    for tj in range(blockDim.y):
                        for ti in range(blockDim.x):
                            threadIdx = ThreadIdx(ti, tj, tk)
                            kernel(blockIdx, threadIdx)

    ic(B)
    ic(C)
    ic(D)
    ic(E)