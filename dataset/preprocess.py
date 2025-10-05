import os.path as osp

from tqdm import tqdm
from datetime import datetime

import shapely
import pandas as pd
import numpy as np
from geopy.distance import geodesic
from math import atan2, degrees

from config.config import Config
from utils.tools import pload
from dataset.grid_space import GridSpace
from dataset.Date2Vec import Date2vec
import torch


def generate_spatial_features(src, cs):
    tgt = []

    for i in range(1, len(src)):
        coord1 = src[i - 1]
        coord2 = src[i]
        coord1[0], coord1[1] = coord1[1], coord1[0]
        coord2[0], coord2[1] = coord2[1], coord2[0]

        distance = geodesic(coord1, coord2).kilometers

        coord1[0], coord1[1] = coord1[1], coord1[0]
        coord2[0], coord2[1] = coord2[1], coord2[0]

        bearing = atan2(coord2[1] - coord1[1], coord2[0] - coord1[0])
        bearing = (degrees(bearing) + 360) % 360 / 360

        x = (coord2[0] - cs.x_min) / (cs.x_max - cs.x_min)
        y = (coord2[1] - cs.y_min) / (cs.y_max - cs.y_min)
        tgt.append([x, y, distance, bearing])

    x = (src[0][0] - cs.x_min) / (cs.x_max - cs.x_min)
    y = (src[0][1] - cs.y_min) / (cs.y_max - cs.y_min)
    tgt.insert(0, [x, y, 0.0, 0.0])
    return tgt


class Preprocess:
    def __init__(self):
        self.path = 'data/{}'.format(Config.dataset)

        self.edge_rn_path = osp.join(self.path, 'rn/edge_rn.csv')
        self.edge_path = osp.join(self.path, 'rn/edge.csv')

        self.edge_rn = pd.read_csv(self.edge_rn_path)
        self.edge = pd.read_csv(self.edge_path)

        x_min, y_min = Config.min_lon, Config.min_lat
        x_max, y_max = Config.max_lon, Config.max_lat

        self.gs = GridSpace(Config.grid_size, Config.grid_size, x_min, y_min, x_max, y_max)
        Config.grid_num = self.gs.grid_num

        self.d2v = Date2vec(Config.hidden_emb_dim, model_path=f'./dataset/d2v_{Config.hidden_emb_dim}d.pt')

        # lazy load cache
        self._road_graph = None
        self._road_feature = None
        self._grid_image = None
        self._train_traj = None
        self._eval_traj = None
        self._test_traj = None

    def get_road_graph(self):
        if self._road_graph is None:
            self._road_graph = np.array([self.edge_rn.from_edge_id, self.edge_rn.to_edge_id])
        return self._road_graph

    def get_road_feature(self):
        if self._road_feature is None:
            path = osp.join(self.path, 'road_feature.npy')
            if osp.exists(path):
                self._road_feature = np.load(path)
            else:
                self._road_feature = self._generate_road_feature()
        return self._road_feature

    def get_grid_image(self):
        if self._grid_image is None:
            path = osp.join(self.path, 'grid_image.npy')
            if osp.exists(path):
                self._grid_image = np.load(path)
            else:
                if self.get_train_traj() is None:
                    raise RuntimeError("Train traj must be loaded to generate grid image")
                self._grid_image = self._construct_grid_image(self._train_traj)
        return self._grid_image

    def get_train_traj(self):
        if self._train_traj is None:
            path = osp.join(self.path, 'traj_train.pkl')
            self._train_traj = pload(path) if osp.exists(path) else None
        return self._train_traj

    def get_eval_traj(self):
        if self._eval_traj is None:
            path = osp.join(self.path, 'traj_eval.pkl')
            self._eval_traj = pload(path) if osp.exists(path) else None
        return self._eval_traj

    def get_test_traj(self):
        if self._test_traj is None:
            path = osp.join(self.path, 'traj_test.pkl')
            self._test_traj = pload(path) if osp.exists(path) else None
        return self._test_traj

    def _generate_road_feature(self):
        def normalization(data):
            return (data - np.min(data)) / (np.max(data) - np.min(data))

        speed = normalization(self.edge['speed_kph'].fillna(0).to_numpy())
        travel_time = normalization(self.edge['travel_time'].fillna(0).to_numpy())
        bearing = normalization(self.edge['bearing'].fillna(0).to_numpy())
        length = normalization(self.edge['length'].fillna(0).to_numpy())
        out_degree = normalization(self.edge['out_degree'].fillna(0).to_numpy())
        in_degree = normalization(self.edge['in_degree'].fillna(0).to_numpy())
        highway_type = pd.get_dummies(self.edge['highway_type']).to_numpy()

        feature = np.concatenate([
            length[:, None], speed[:, None], travel_time[:, None],
            bearing[:, None], out_degree[:, None], in_degree[:, None],
            highway_type
        ], axis=1)

        np.save(osp.join(self.path, 'road_feature.npy'), feature)
        return feature

    def _construct_grid_image(self, trajs):
        gps_traj = list(trajs['gps_list'])
        traffic_image = np.zeros((self.gs.x_size, self.gs.y_size))
        for traj in gps_traj:
            for x, y in traj:
                p_x, p_y = self.gs.get_xyidx_by_point(x, y)
                traffic_image[p_x, p_y] += 1
        traffic_image = (traffic_image - np.min(traffic_image)) / (np.max(traffic_image) - np.min(traffic_image))

        x_image = np.zeros((self.gs.x_size, self.gs.y_size))
        y_image = np.zeros((self.gs.x_size, self.gs.y_size))
        for i_x in range(self.gs.x_size):
            for i_y in range(self.gs.y_size):
                c_x, c_y = self.gs.get_center_point(i_x, i_y)
                x_image[i_x, i_y] = c_x
                y_image[i_x, i_y] = c_y
        x_max = self.gs.x_size * self.gs.x_unit + self.gs.x_min
        y_max = self.gs.y_size * self.gs.y_unit + self.gs.y_min
        x_image = (x_image - self.gs.x_min) / (x_max - self.gs.x_min)
        y_image = (y_image - self.gs.y_min) / (y_max - self.gs.y_min)

        image = np.concatenate([
            x_image[:, :, None],
            y_image[:, :, None],
            traffic_image[:, :, None]
        ], axis=-1)

        np.save(osp.join(self.path, 'grid_image.npy'), image)
        return image
