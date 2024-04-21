import numpy as np

from sklearn.utils.random import sample_without_replacement
from pathlib import Path

BATCHES_COUNT = 100


def main():
    data_path = next(Path("..").glob("*.npy"))
    dataset = np.load(str(data_path))

    arrange = np.linspace(1e-2 * dataset.shape[0], dataset.shape[0], BATCHES_COUNT, endpoint=True).astype(np.int32)
    for index, batch_size in enumerate(arrange[::-1]):
        batch_indices = sample_without_replacement(dataset.shape[0], batch_size)
        np.save(f"../batches/batch-{index + 1}.npy", batch_indices)


if __name__ == "__main__":
    main()
