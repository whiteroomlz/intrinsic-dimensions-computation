import re
import os
import scipy
import numpy as np
import pandas as pd

from mst_clustering.cpp_adapters import MstBuilder, SpanningForest, DistanceMeasure
from sklearn.neighbors import kneighbors_graph
from scipy.sparse.csgraph import dijkstra
from scipy.special import comb
from pathlib import Path
from scipy import stats
from tqdm import tqdm
from enum import Enum


class BqyIntrinsicDimensionEstimator(object):
    candidate_dimensions: np.ndarray
    sample_size: int
    samples_count: int
    workers_count: int
    batch_weight_in_gb: float

    knn_candidate_dim_means: np.ndarray
    knn_candidate_dim_vars: np.ndarray

    mst_candidate_dim_means: np.ndarray
    mst_candidate_dim_vars: np.ndarray

    soi_candidate_dim_means: np.ndarray
    soi_candidate_dim_vars: np.ndarray

    class StatType(Enum):
        KNN = 0
        MST = 1
        SOI = 2

    def __init__(self,
                 candidate_dimensions,
                 sample_size=1000,
                 samples_count=100,
                 workers_count=1,
                 batch_weight_in_gb=1.0,
                 knn_k_neighbours=1, knn_path_limit=1,
                 knn_candidate_dim_means=None, knn_candidate_dim_vars=None,
                 mst_candidate_dim_means=None, mst_candidate_dim_vars=None,
                 soi_k_neighbours=1,
                 soi_candidate_dim_means=None, soi_candidate_dim_vars=None,
                 stat_types=[StatType.MST]):

        self.candidate_dimensions = candidate_dimensions
        self.sample_size = sample_size
        self.samples_count = samples_count
        self.workers_count = workers_count
        self.batch_weight_in_gb = batch_weight_in_gb
        self.knn_k_neighbours = knn_k_neighbours
        self.knn_path_limit = knn_path_limit
        self.soi_k_neighbours = soi_k_neighbours

        if self.StatType.KNN in stat_types:
            if not (knn_candidate_dim_means is None or knn_candidate_dim_vars is None):
                self.knn_candidate_dim_means = knn_candidate_dim_means
                self.knn_candidate_dim_vars = knn_candidate_dim_vars
            else:
                self.init_knn_stat(knn_k_neighbours, knn_path_limit)

        if self.StatType.MST in stat_types:
            if not (mst_candidate_dim_means is None or mst_candidate_dim_vars is None):
                self.mst_candidate_dim_means = mst_candidate_dim_means
                self.mst_candidate_dim_vars = mst_candidate_dim_vars
            else:
                self.init_mst_stat()

        if self.StatType.SOI in stat_types:
            if not (soi_candidate_dim_means is None or soi_candidate_dim_vars is None):
                self.soi_candidate_dim_means = soi_candidate_dim_means
                self.soi_candidate_dim_vars = soi_candidate_dim_vars
            else:
                self.init_soi_stat(soi_k_neighbours)

    def init_knn_stat(self, knn_k_neighbours, knn_path_limit):
        self.knn_candidate_dim_means = np.zeros(self.candidate_dimensions.size)
        self.knn_candidate_dim_vars = np.zeros(self.candidate_dimensions.size)

        for candidate_dim_index, candidate_dimension in tqdm(
                enumerate(self.candidate_dimensions),
                total=len(self.candidate_dimensions),
                desc="Computing KNN candidate-dimension statistics..."):

            knn_candidate_dim_stats = np.zeros(self.samples_count)

            for sample_index in np.arange(self.samples_count):
                sample = np.random.uniform(size=(self.sample_size, candidate_dimension))
                knn_graph = kneighbors_graph(
                    sample, knn_k_neighbours, n_jobs=self.workers_count, mode='connectivity', include_self=False
                )
                knn_candidate_dim_stat = self.compute_knn_stat(knn_graph, knn_path_limit, self.batch_weight_in_gb)
                knn_candidate_dim_stats[sample_index] = knn_candidate_dim_stat

            self.knn_candidate_dim_means[candidate_dim_index] = np.mean(knn_candidate_dim_stats)
            self.knn_candidate_dim_vars[candidate_dim_index] = np.var(knn_candidate_dim_stats) * self.sample_size

    def init_mst_stat(self):
        self.mst_candidate_dim_means = np.zeros(self.candidate_dimensions.size)
        self.mst_candidate_dim_vars = np.zeros(self.candidate_dimensions.size)

        for candidate_dim_index, candidate_dimension in tqdm(
                enumerate(self.candidate_dimensions),
                total=len(self.candidate_dimensions),
                desc="Computing MST candidate-dimension statistics..."):

            mst_candidate_dim_stats = np.zeros(self.samples_count)

            for sample_index in np.arange(self.samples_count):
                sample = np.random.uniform(size=(self.sample_size, candidate_dimension))
                mst = MstBuilder(sample.tolist()).build(
                    workers_count=self.workers_count,
                    distance_measure=DistanceMeasure.EUCLIDEAN
                )
                mst_candidate_dim_stat = self.compute_mst_stat(mst)[0]
                mst_candidate_dim_stats[sample_index] = mst_candidate_dim_stat

            self.mst_candidate_dim_means[candidate_dim_index] = np.mean(mst_candidate_dim_stats)
            self.mst_candidate_dim_vars[candidate_dim_index] = np.var(mst_candidate_dim_stats) * self.sample_size

    def init_soi_stat(self, soi_k_neighbours):
        self.soi_candidate_dim_means = np.zeros(self.candidate_dimensions.size)
        self.soi_candidate_dim_vars = np.zeros(self.candidate_dimensions.size)

        for candidate_dim_index, candidate_dimension in tqdm(
                enumerate(self.candidate_dimensions),
                total=len(self.candidate_dimensions),
                desc="Computing SOI candidate-dimension statistics..."):

            soi_candidate_dim_stats = np.zeros(self.samples_count)

            for sample_index in np.arange(self.samples_count):
                sample = np.random.uniform(size=(self.sample_size, candidate_dimension))
                knn_graph = kneighbors_graph(
                    sample, soi_k_neighbours, n_jobs=self.workers_count, mode='connectivity', include_self=False
                )
                soi_candidate_dim_stat = self.compute_soi_stat(knn_graph)
                soi_candidate_dim_stats[sample_index] = soi_candidate_dim_stat

            self.soi_candidate_dim_means[candidate_dim_index] = np.mean(soi_candidate_dim_stats)
            self.soi_candidate_dim_vars[candidate_dim_index] = np.var(soi_candidate_dim_stats) * self.sample_size

    def compute_probabilities(self, stat, sample_size, stat_type):
        if stat_type is self.StatType.KNN:
            candidate_dim_means, candidate_dim_vars = self.knn_candidate_dim_means, self.knn_candidate_dim_vars
        elif stat_type is self.StatType.MST:
            candidate_dim_means, candidate_dim_vars = self.mst_candidate_dim_means, self.mst_candidate_dim_vars
        elif stat_type is self.StatType.SOI:
            candidate_dim_means, candidate_dim_vars = self.soi_candidate_dim_means, self.soi_candidate_dim_vars

        probabilities = np.fromiter(map(
            lambda mean_var: stats.norm(loc=mean_var[0], scale=np.sqrt(mean_var[1] / sample_size)).pdf(stat),
            zip(candidate_dim_means[:self.candidate_dimensions.size],
                candidate_dim_vars[:self.candidate_dimensions.size])
        ), dtype=np.float64)

        error_message = "The variances are too small."
        assert not np.sum(probabilities) == 0, error_message

        return probabilities

    def estimate_by_knn(self, knn_graph: scipy.sparse._csr.csr_matrix, round_result=True):
        vertex_count = knn_graph.shape[0]
        knn_stat = self.compute_knn_stat(knn_graph, self.knn_path_limit, self.workers_count)

        return self.estimate_by_stat(knn_stat, vertex_count, self.StatType.KNN, round_result)

    def estimate_by_mst(self, mst, round_result=True):
        mst_stat, vertex_count = self.compute_mst_stat(mst)

        return self.estimate_by_stat(mst_stat, vertex_count, self.StatType.MST, round_result)

    def estimate_by_soi(self, knn_graph: scipy.sparse._csr.csr_matrix, round_result=True):
        vertex_count = knn_graph.shape[0]
        soi_stat = self.compute_soi_stat(knn_graph)

        return self.estimate_by_stat(soi_stat, vertex_count, self.StatType.SOI, round_result)

    def estimate_by_stat(self, stat, vertex_count, stat_type, round_result=True):
        probabilities = self.compute_probabilities(stat, vertex_count, stat_type)

        return self.estimate(probabilities, round_result)

    def estimate(self, probabilities, round_result=True):
        mathematical_expectation = np.sum(self.candidate_dimensions * probabilities) / np.sum(probabilities)
        intrinsic_dimension = np.round(mathematical_expectation) if round_result else mathematical_expectation

        return intrinsic_dimension

    @staticmethod
    def compute_knn_stat(knn_graph: scipy.sparse._csr.csr_matrix, path_limit, batch_weight_in_gb=1.0) -> float:
        batch_size = batch_weight_in_gb * (1024 ** 3) // 8
        batches_count = knn_graph.shape[0] // (batch_size // knn_graph.shape[0])
        if batches_count == 0:
            batches_count = 1

        all_indices = np.arange(knn_graph.shape[0])

        knn_counts = np.zeros(knn_graph.shape[0])
        for batch_indices in np.array_split(all_indices, batches_count):
            distances = dijkstra(knn_graph, directed=False, limit=path_limit, indices=batch_indices)
            knn_counts[batch_indices] = np.sum(distances != np.inf, axis=1)

        knn_stat = np.mean(knn_counts)

        return knn_stat

    @staticmethod
    def compute_mst_stat(mst: SpanningForest) -> float:
        assert mst.is_spanning_tree, "Spanning forest contains multiple connectivity components."

        edges = np.array(mst.get_tree_edges(*mst.get_roots()))
        extract_nodes = np.vectorize(lambda edge: (edge.first_node, edge.second_node))
        nodes = np.array(extract_nodes(edges))

        vertex_count = edges.size + 1
        vertex_degrees = np.unique(nodes, return_counts=True)[1]

        mst_stat = np.sum(vertex_degrees ** 2) / vertex_count

        return mst_stat, vertex_count

    @staticmethod
    def compute_soi_stat(knn_graph: scipy.sparse._csr.csr_matrix) -> float:
        cardinalities = knn_graph.sum(axis=0)
        soi_stat = np.sum(comb(cardinalities, 2)) / knn_graph.shape[0]

        return soi_stat


def main():
    precomputed_mst_files = list(Path('../minimum-spanning-trees').glob('*.npy'))
    precomputed_mst_files_numbers = list(map(
        lambda mst_file: re.sub(r'batch-(\d+)-mst', r'\1', mst_file.stem), precomputed_mst_files
    ))
    sizes = np.fromiter(map(
        lambda number: np.load(f'../batches/batch-{number}.npy').shape[0], precomputed_mst_files_numbers
    ), dtype=np.float64)

    candidate_dimensions = np.arange(2, 17)

    stats_path = f'../out/bqy-mst-stats.npy'
    if os.path.isfile(stats_path):
        mst_stats = np.load(stats_path)
    else:
        mst_stats = np.zeros(sizes.size)
        for index, mst_file in tqdm(enumerate(precomputed_mst_files), desc='Computing stats...', total=sizes.size):
            mst = SpanningForest.load(mst_file)
            mst_stats[index] = BqyIntrinsicDimensionEstimator.compute_mst_stat(mst)[0]
        np.save(stats_path, mst_stats)

    means_path = '../bqy-stats/mst-estimator-means-{}-2-17.npy'
    vars_path = '../bqy-stats/mst-estimator-vars-{}-2-17.npy'
    precomputed_stats_sizes = [10000, 100000, 1000000]
    mst_candidate_dim_means = [np.load(means_path.format(size)) for size in precomputed_stats_sizes]
    mst_candidate_dim_vars = [np.load(vars_path.format(size)) for size in precomputed_stats_sizes]

    estimated_dims = np.zeros(sizes.size)
    for index, size in tqdm(enumerate(sizes), desc='Estimating...'):
        if size < precomputed_stats_sizes[-1]:
            precomputed_stats_size_index = np.arange(len(precomputed_stats_sizes))[size < precomputed_stats_sizes][0]
        else:
            precomputed_stats_size_index = -1

        estimator = BqyIntrinsicDimensionEstimator(
            candidate_dimensions,
            mst_candidate_dim_means=mst_candidate_dim_means[precomputed_stats_size_index],
            mst_candidate_dim_vars=mst_candidate_dim_vars[precomputed_stats_size_index],
            stat_types=[BqyIntrinsicDimensionEstimator.StatType.MST]
        )
        try:
            estimated_dims[index] = estimator.estimate_by_stat(
                mst_stats[index], sizes[index], estimator.StatType.MST, round_result=False
            )
        except AssertionError as _:
            continue

    df = pd.DataFrame({
        'estimated_dim': estimated_dims,
        'batch_size': sizes
    })
    df.to_excel('../out/bqy-intrinsic-dimensions.xlsx')


if __name__ == '__main__':
    main()
