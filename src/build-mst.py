import argparse
import numpy as np

from mst_clustering.cpp_adapters import MstBuilder, DistanceMeasure
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('batch_path', type=str)
args = parser.parse_args()


def main():
    data_path = next(Path("..").glob("*.npy"))
    dataset = np.load(str(data_path))

    batch_indices = np.load(args.batch_path)
    batch = dataset[batch_indices, :]

    min_vals = dataset.min(axis=0)
    max_vals = dataset.max(axis=0)
    normalized_batch = (batch - min_vals) / (max_vals - min_vals)

    builder = MstBuilder(normalized_batch.tolist())
    mst = builder.build(4, DistanceMeasure.EUCLIDEAN)

    mst.save(f"../minimum-spanning-trees/{Path(args.batch_path).stem}-mst")


if __name__ == '__main__':
    main()
