import yaml
import os
import pdb
import numpy as np
import tensorflow as tf

from select_knn_op import SelectKnn
from slicing_knn_op import SlicingKnn
from binned_select_knn_op import BinnedSelectKnn
from select_mod_knn_op import SelectModKnn
from accknn_op import AccumulateKnn, AccumulateLinKnn
from local_cluster_op import LocalCluster
from local_group_op import LocalGroup
from local_distance_op import LocalDistance
from neighbour_covariance_op import NeighbourCovariance as NeighbourCovarianceOp
from assign_condensate_op import BuildAndAssignCondensatesBinned as ba_cond
from pseudo_rs_op import create_prs_indices, revert_prs

# just for the moment
#### helper###
from Initializers import EyeInitializer
from oc_helper_ops import SelectWithDefault
from baseModules import LayerWithMetrics


class RaggedEGCN(tf.keras.layers.Layer):
    def __init__(
        self,
        n_neighbours: int,
        n_dimensions: int,
        n_filters: int,
        n_propagate: int,
        return_self=True,
        sumwnorm=False,
        feature_activation="selu",
        use_approximate_knn=False,
        coord_initialiser_noise=1e-2,
        use_dynamic_knn=True,
        debug=False,
        n_knn_bins=None,
        _promptnames=None,  # compatibility, does nothing
        record_metrics=False,  # compatibility, does nothing
        **kwargs
    ):

        super(RaggedEGCN, self).__init__(**kwargs)

        # n_neighbours += 1  # includes the 'self' vertex
        assert n_neighbours > 1
        assert not use_approximate_knn  # not needed anymore. Exact one is faster by now

        self.n_neighbours = n_neighbours
        self.n_dimensions = n_dimensions
        self.n_filters = n_filters
        self.return_self = return_self
        self.sumwnorm = sumwnorm
        self.feature_activation = "selu"
        self.use_approximate_knn = use_approximate_knn
        self.use_dynamic_knn = use_dynamic_knn
        self.debug = debug
        self.n_knn_bins = n_knn_bins

        self.n_propagate = 128
        self.n_prop_total = 2 * self.n_propagate

        with tf.name_scope(self.name + "/1/"):
            self.edge_mlp = tf.keras.layers.Dense(128, activation=feature_activation)
        with tf.name_scope(self.name + "/2/"):
            self.edge_mlp2 = tf.keras.layers.Dense(128, activation=feature_activation)

        with tf.name_scope(self.name + "/3/"):
            self.coord_mlp = tf.keras.layers.Dense(128, activation=feature_activation)
        with tf.name_scope(self.name + "/4/"):
            self.coord_mlp2 = tf.keras.layers.Dense(
                1, kernel_initializer="glorot_uniform"
            )

        with tf.name_scope(self.name + "/5/"):
            self.node_mlp = tf.keras.layers.Dense(
                128, activation=feature_activation
            )  # changed to relu

        with tf.name_scope(self.name + "/6/"):
            self.node_mlp2 = tf.keras.layers.Dense(128)  # changed to relu

    def build(self, input_shapes):
        input_shape = input_shapes[0]
        with tf.name_scope(self.name + "/1/"):
            self.edge_mlp.build((None, 2 * input_shape[1] + 1))
        with tf.name_scope(self.name + "/2/"):
            self.edge_mlp2.build((None, 128))

        with tf.name_scope(self.name + "/3/"):
            self.coord_mlp.build((None, 128))

        with tf.name_scope(self.name + "/4/"):
            self.coord_mlp2.build((None, 128))

        with tf.name_scope(self.name + "/5/"):
            self.node_mlp.build((None, 128 + input_shape[1]))
        with tf.name_scope(self.name + "/6/"):
            self.node_mlp2.build((None, 128))

        super(RaggedEGCN, self).build(input_shape)

    def priv_call(self, inputs, coordinates, training=None):
        h = inputs[0]
        shape_features = h.shape[1]
        row_splits = inputs[1]
        (
            neighbour_indices,
            distancesq,
            sidx,
            sdist,
            coord_diff,
        ) = self.compute_neighbours_and_distancesq(coordinates, row_splits, training)
        neighbour_indices = tf.reshape(
            neighbour_indices, [-1, self.n_neighbours]
        )  # for proper output shape for keras
        radial0 = tf.reshape(distancesq, [-1, 1])
        h = SelectWithDefault(sidx, h, 0.0)
        h = tf.reshape(h, [-1, self.n_neighbours + 1, shape_features])
        h_neig = tf.reshape(h[:, 1:, :], [-1, shape_features])  # B x Kn x N_features
        h_node = tf.reshape(h[:, 0:1, :], [-1, 1, shape_features])
        h_node = tf.tile(h_node, [1, self.n_neighbours, 1])
        h_node = tf.reshape(h_node, [-1, shape_features])
        # edge_feature = tf.cat((radial0,))
        edge_feature = tf.concat((radial0, h_node, h_neig), axis=1)  # [NxK,2H+1]
        edge_feature = self.edge_mlp(edge_feature)  # (BxNk)x128
        edge_feature = self.edge_mlp2(edge_feature)
        cord_mlp = self.coord_mlp(edge_feature)
        cord_mlp2 = self.coord_mlp2(cord_mlp)
        cord_mlp2 = tf.reshape(cord_mlp2, [-1, self.n_neighbours, 1])
        cord_mlp2 = tf.tile(cord_mlp2, [1, 1, 3])
        trans = coord_diff * cord_mlp2
        trans = tf.reduce_mean(trans, axis=1)
        coods_new = coordinates + trans
        edge_feature = tf.reshape(edge_feature, [-1, self.n_neighbours, 128])
        edge_feature = tf.reduce_sum(edge_feature, axis=1)
        h_node_ = tf.reshape(h[:, 0:1, :], [-1, shape_features])
        agg = tf.concat((edge_feature, h_node_), axis=1)
        h = self.node_mlp(agg)  # [None, h input+128]
        h = self.node_mlp2(h)
        return h, coods_new, neighbour_indices, distancesq

    def call(self, inputs, training):
        return self.priv_call(inputs, training)

    # def compute_output_shape(self, input_shapes):
    #     if self.return_self:
    #         return (
    #             (input_shapes[0][0], 2 * self.n_filters),
    #             (input_shapes[0][0], self.n_dimensions),
    #             (input_shapes[0][0], self.n_neighbours + 1),
    #             (input_shapes[0][0], self.n_neighbours + 1),
    #         )
    #     else:
    #         return (
    #             (input_shapes[0][0], 2 * self.n_filters),
    #             (input_shapes[0][0], self.n_dimensions),
    #             (input_shapes[0][0], self.n_neighbours),
    #             (input_shapes[0][0], self.n_neighbours),
    #         )

    def compute_neighbours_and_distancesq(self, coordinates, row_splits, training):

        idx, dist, coord_diff = BinnedSelectKnn(
            self.n_neighbours + 1,
            coordinates,
            row_splits,
            max_radius=-1.0,
            tf_compatible=False,
            n_bins=self.n_knn_bins,
            name=self.name,
            return_coord_difs=True,
        )
        idx = tf.reshape(idx, [-1, self.n_neighbours + 1])
        dist = tf.reshape(dist, [-1, self.n_neighbours + 1])
        coord_diff = tf.reshape(coord_diff, [-1, self.n_neighbours + 1, 3])

        dist = tf.where(idx < 0, 0.0, dist)

        if self.return_self:
            return idx[:, 1:], dist[:, 1:], idx, dist, coord_diff[:, 1:, :]
        return idx[:, 1:], dist[:, 1:], None, None, coord_diff[:, 1:, :]

    def collect_neighbours(self, features, neighbour_indices, distancesq):
        f = None
        if self.sumwnorm:
            f, _ = AccumulateKnnSumw(
                10.0 * distancesq, features, neighbour_indices, mean_and_max=True
            )
        else:
            f, _ = AccumulateKnn(
                10.0 * distancesq, features, neighbour_indices, mean_and_max=True
            )
        return f

    def get_config(self):
        config = {
            "n_neighbours": self.n_neighbours,
            "n_dimensions": self.n_dimensions,
            "n_filters": self.n_filters,
            "n_propagate": self.n_propagate,
            "return_self": self.return_self,
            "sumwnorm": self.sumwnorm,
            "feature_activation": self.feature_activation,
            "use_approximate_knn": self.use_approximate_knn,
            "n_knn_bins": self.n_knn_bins,
        }
        base_config = super(RaggedEGCN, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
