import warnings
import numpy as np

np.warnings = warnings

import re
import math
import scipy
import sklearn
import os.path
import numpy as np
import pandas as pd
import seaborn as sns
import multiprocessing
import plotly.express as px
import matplotlib.pyplot as plt
import plotly.graph_objects as go

from mst_clustering.cpp_adapters import MstBuilder, SpanningForest, DistanceMeasure
from mst_clustering.clustering_models import ZahnModel
from matplotlib.animation import FuncAnimation
from sklearn.neighbors import kneighbors_graph
from mst_clustering.pipeline import Pipeline
from scipy.sparse.csgraph import dijkstra
from plotly.subplots import make_subplots
from scipy.spatial import KDTree
from scipy.special import comb
from itertools import repeat
from sklearn import datasets
from pathlib import Path
from scipy import stats
from tqdm import tqdm
from enum import Enum

sns.set_style('whitegrid')

multiprocessing.freeze_support()


class SchweinhartIntrinsicDimensionEstimator(object):
    msts: list
    sample_sizes: np.ndarray
    msts_edges_weights: list

    def __init__(self, msts, sample_sizes, e_stats_dict=None, msts_edges_weights=None, save_stats=True):
        self.msts = msts
        self.sample_sizes = sample_sizes
        if e_stats_dict is not None:
            self.e_stats_dict = e_stats_dict
        else:
            self.e_stats_dict = dict()

        self.msts_edges_weights = msts_edges_weights

        self.save_stats = save_stats

    def estimate(self, alpha):
        e_stat = self.get_e_stat(alpha)

        X = np.hstack((np.log(self.sample_sizes)[:, np.newaxis], np.ones((len(self.msts), 1))))
        y = np.log(e_stat)
        w = np.linalg.pinv(X) @ y

        intrinsic_dimension = alpha / (1 - w[0])

        return intrinsic_dimension

    def get_dim_conf_interval(self, alpha, conf_coefficient):
        e_stat = self.get_e_stat(alpha)

        X = np.hstack((np.log(self.sample_sizes)[:, np.newaxis], np.ones((len(self.msts), 1))))
        y = np.log(e_stat)
        w = np.linalg.pinv(X) @ y

        # Якобиан.
        J = np.hstack((
            (np.exp(w[1]) * np.log(self.sample_sizes) * self.sample_sizes ** w[0])[:, np.newaxis],
            (self.sample_sizes ** w[0])[:, np.newaxis]
        ))

        t_quantile = scipy.stats.t(X.shape[0] - X.shape[1]).ppf(1 - conf_coefficient / 2)
        noise_quad = np.sum((np.exp(y) - np.exp(X @ w)) ** 2) / (X.shape[0] - X.shape[1])
        # Нас интересует доверительный интервал для нулевого веса, в который входит размерность.
        delta = t_quantile * np.sqrt(noise_quad) * np.sqrt(np.linalg.inv(J.T @ J)[0, 0])

        w0_lower_bound = w[0] - delta
        w0_upper_bound = w[0] + delta

        intrinsic_dim_lower_bound = alpha / (1 - min([w0_lower_bound, 1]))
        intrinsic_dim_upper_bound = alpha / (1 - min([w0_upper_bound, 1]))

        conf_interval = np.array([intrinsic_dim_lower_bound, intrinsic_dim_upper_bound])

        return conf_interval

    def get_regression_conf_intervals(self, alpha, conf_coefficient):
        e_stat = self.get_e_stat(alpha)

        X = np.hstack((np.log(self.sample_sizes)[:, np.newaxis], np.ones((len(self.msts), 1))))
        y = np.log(e_stat)
        w = np.linalg.pinv(X) @ y

        # Якобиан.
        J = np.hstack((
            (np.exp(w[1]) * np.log(self.sample_sizes) * self.sample_sizes ** w[0])[:, np.newaxis],
            (self.sample_sizes ** w[0])[:, np.newaxis]
        ))

        t_quantile = scipy.stats.t(X.shape[0] - X.shape[1]).ppf(1 - conf_coefficient / 2)
        noise_quad = np.sum((np.exp(y) - np.exp(X @ w)) ** 2) / (X.shape[0] - X.shape[1])

        deltas = np.zeros(X.shape[0])
        for index in np.arange(X.shape[0]):
            deltas[index] = t_quantile * np.sqrt(noise_quad) * \
                            np.sqrt(J[index] @ np.linalg.inv(J.T @ J) @ J[index].T)

        lower_bounds = np.exp(X @ w) - deltas
        upper_bounds = np.exp(X @ w) + deltas

        conf_intervals = np.hstack([lower_bounds[:, np.newaxis], upper_bounds[:, np.newaxis]])

        return conf_intervals, e_stat

    def get_regression_pred_intervals(self, alpha, conf_coefficient):
        e_stat = self.get_e_stat(alpha)

        X = np.hstack((np.log(self.sample_sizes)[:, np.newaxis], np.ones((len(self.msts), 1))))
        y = np.log(e_stat)
        w = np.linalg.pinv(X) @ y

        # Якобиан.
        J = np.hstack((
            (np.exp(w[1]) * np.log(self.sample_sizes) * self.sample_sizes ** w[0])[:, np.newaxis],
            (self.sample_sizes ** w[0])[:, np.newaxis]
        ))

        t_quantile = scipy.stats.t(X.shape[0] - X.shape[1]).ppf(1 - conf_coefficient / 2)
        noise_quad = np.sum((np.exp(y) - np.exp(X @ w)) ** 2) / (X.shape[0] - X.shape[1])

        deltas = np.zeros(X.shape[0])
        for index in np.arange(X.shape[0]):
            deltas[index] = t_quantile * np.sqrt(noise_quad) * \
                            np.sqrt(1 + J[index] @ np.linalg.inv(J.T @ J) @ J[index].T)

        lower_bounds = np.exp(X @ w) - deltas
        upper_bounds = np.exp(X @ w) + deltas

        pred_intervals = np.hstack([lower_bounds[:, np.newaxis], upper_bounds[:, np.newaxis]])

        return pred_intervals, e_stat

    def get_e_stat(self, alpha):
        if alpha in self.e_stats_dict.keys() and self.e_stats_dict[alpha].size == len(self.msts):
            e_stat = self.e_stats_dict[alpha]
        else:
            if self.msts_edges_weights is None:
                self.msts_edges_weights = list()
                for mst in tqdm(self.msts, desc='Extracting weights of edges...'):
                    self.msts_edges_weights.append(self.get_mst_weights(mst))

            e_stat = np.zeros(len(self.msts))
            for index, mst_weights in enumerate(self.msts_edges_weights):
                e_stat[index] = self.compute_e_stat(mst_weights, alpha)

            if self.save_stats:
                self.e_stats_dict[alpha] = e_stat

        return e_stat

    @staticmethod
    def get_mst_weights(mst):
        edges = np.array(mst.get_tree_edges(*mst.get_roots()))
        extract_weights = np.vectorize(lambda edge: edge.weight)
        weights = np.array(extract_weights(edges))

        return weights

    @staticmethod
    def compute_e_stat(mst_weights, alpha):
        e_stat = np.sum(mst_weights ** alpha)

        return e_stat


def remove_outliers(tree_path, indices_path, bad_indices, out_path):
    tree = SpanningForest.load(tree_path)
    indices = np.load(indices_path)

    edges = np.array(tree.get_tree_edges(*tree.get_roots()))

    for edge in edges:
        first_node_index = indices[edge.first_node]
        second_node_index = indices[edge.second_node]

        if ((first_node_index, second_node_index) in bad_indices) or (
                (second_node_index, first_node_index) in bad_indices
        ):
            tree.remove_edge(edge.first_node, edge.second_node)

    idx = 0
    for root in tree.get_roots():
        if tree.get_tree_size(root) < 2:
            continue
        new_nodes = tree.get_tree_nodes(root)
        new_indices = indices[new_nodes]
        new_edges = tree.get_tree_edges(root)
        new_weights = np.array(extract_weights(new_edges))
        new_indices_path = f"{out_path}/indices/{tree_path.stem}-{idx}"
        new_weights_path = f"{out_path}/weights/{tree_path.stem}-{idx}"
        np.save(new_indices_path, new_indices)
        np.save(new_weights_path, new_weights)
        idx += 1

if __name__ == "__main__":
    precomputed_mst_files = list(Path('../minimum-spanning-trees/').glob('*.npy'))
    precomputed_mst_files_numbers = list(map(
        lambda mst_file: re.sub(r'batch-(\d+)-mst', r'\1', mst_file.stem), precomputed_mst_files
    ))
    sizes = np.fromiter(map(
        lambda number: np.load(f'../batches/batch-{number}.npy').shape[0], precomputed_mst_files_numbers
    ), dtype=np.float64)

    sorted_indices = np.argsort(sizes)
    precomputed_mst_files = np.array(precomputed_mst_files)[sorted_indices]
    sizes = sizes[sorted_indices]
    
    indices_files = np.fromiter(map(
        lambda number: f'../batches/batch-{number}.npy', precomputed_mst_files_numbers
    ), dtype=object)
    indices_files = indices_files[sorted_indices]

    out_path = '../remove_outliers/'

    tree_path = precomputed_mst_files[-1]
    indices_path = indices_files[-1]

    biggest_tree = SpanningForest.load(tree_path)
    indices = np.load(indices_path)

    edges = np.array(biggest_tree.get_tree_edges(*biggest_tree.get_roots()))
    extract_weights = np.vectorize(lambda edge: edge.weight)
    weights = np.array(extract_weights(edges))

    flags = np.where(weights > np.quantile(weights, .9999), 1, 0)
    huge_weights_indices = np.argsort(flags)[-np.sum(flags):]
    huge_edges = edges[huge_weights_indices]

    bad_indices_first = list(map(lambda i: indices[i.first_node], huge_edges))
    bad_indices_second = list(map(lambda i: indices[i.second_node], huge_edges))
    bad_indices = set(zip(bad_indices_first, bad_indices_second))

    for tree_path, indices_path in tqdm(zip(precomputed_mst_files, indices_files)):
        remove_outliers(tree_path, indices_path, bad_indices, out_path)
    
    msts_edges_weights = list()
    
    sizes = list()
    for file in Path('../remove_outliers/weights').glob('*.npy'):
        edges_weights_curr = np.load(file)
        msts_edges_weights.append(edges_weights_curr)
        indices_path = f'../remove_outliers/indices/{file.name}'
        indices_curr = np.load(indices_path)
        sizes.append(indices_curr.size)
    sizes = np.array(sizes)

    estimator = SchweinhartIntrinsicDimensionEstimator(
        [None] * len(msts_edges_weights),
        sizes,
        msts_edges_weights=msts_edges_weights,
        save_stats=False
    )

    alphas = np.linspace(1, 10, 100)
    estimated_dims = np.zeros(alphas.size)
    dim_conf_intervals = np.zeros((alphas.size, 2))

    for index, alpha in tqdm(enumerate(alphas), desc='Computing intrinsic dimensions...'):
        estimated_dims[index] = estimator.estimate(alpha)
        dim_conf_intervals[index] = estimator.get_dim_conf_interval(alpha, 0.05)

    df = pd.DataFrame({
        'estimated_dim': estimated_dims,
        'conf_lower_bound': dim_conf_intervals[:, 0],
        'conf_upper_bound': dim_conf_intervals[:, 1],
        'alpha': alphas
    })
    df.to_excel('../out/schweinhart-intrinsic-dimensions-clear.xlsx')
