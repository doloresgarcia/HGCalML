BATCHNORM_OPTIONS = {
    "max_viscosity": 0.999999,  # keep very batchnorm like
    "fluidity_decay": 0.01,
}


# Configuration for training
DENSE_ACTIVATION = "elu"  # layernorm #'elu'
LEARNINGRATE = 1e-4
NBATCH = 120000  # 200000
DENSE_REGULARIZER = None


# record internal metrics every N batches
record_frequency = 20
# plot every M times, metrics were recorded. In other words,
# plotting will happen every M*N batches
plotfrequency = 100


LOSS_OPTIONS = {
    "energy_loss_weight": 0.0,
    "q_min": 3.0,
    "use_average_cc_pos": 0.99,
    "classification_loss_weight": 0.0,  # to make it work0.5,
    "too_much_beta_scale": 0.0,
    "position_loss_weight": 0.0,
    "timing_loss_weight": 0.0,
    "beta_loss_scale": 1.0,  # 2.0
    "implementation": "hinge_full_grad",  #'hinge_manhatten'#'hinge'#old school
}
import wandb

USE_WANDB = True

if USE_WANDB:
    wandb.init(
        project="hgcalml_work_branch",
        tags=["debug", "small_dataset"],
        name="train_1008_logging_pictures",
    )
    wandb.run.log_code(".")

import tensorflow as tf
import globals

if True:  # for testing
    # globals.acc_ops_use_tf_gradients = True
    globals.knn_ops_use_tf_gradients = True

import tensorflow as tf
from tensorflow.keras.layers import Dense, Concatenate, GaussianDropout, Dropout

import training_base_hgcal
from DeepJetCore.training.DeepJet_callbacks import simpleMetricsCallback
from DeepJetCore.DJCLayers import StopGradient
from datastructures import TrainData_PreselectionNanoML

from Layers import RaggedEGCN, RaggedGlobalExchange
from Layers import DistanceWeightedMessagePassing
from Layers import DictModel, SphereActivation, Multi
from Layers import CastRowSplits, PlotCoordinates
from Layers import LLFullObjectCondensation as LLExtendedObjectCondensation
from Layers import ScaledGooeyBatchNorm2, Sqrt
from Layers import LLFillSpace, SphereActivation
from Regularizers import AverageDistanceRegularizer
from model_blocks import create_outputs
from model_blocks import extent_coords_if_needed
from model_blocks import tiny_pc_pool, condition_input
from model_tools import apply_weights_from_path
from callbacks import plotEventDuringTraining, plotClusteringDuringTraining
from callbacks import plotClusterSummary, NanSweeper
from Layers import layernorm, EdgeConvStatic
import os
from argparse import ArgumentParser
from tensorflow.keras.layers import Dense, Concatenate, Add, Reshape, Flatten

from DeepJetCore.DJCLayers import StopGradient

from Layers import (
    RaggedGlobalExchange,
    DistanceWeightedMessagePassing,
    DictModel,
    LLRegulariseGravNetSpace,
)
from Layers import RaggedEGCN, ScaledGooeyBatchNorm2
from Regularizers import AverageDistanceRegularizer
from LossLayers import LLFullObjectCondensation, LLFillSpace
from wandb_callback import wandbCallback
from DebugLayers import PlotCoordinates

from model_blocks import (
    condition_input,
    extent_coords_if_needed,
    create_outputs,
    re_integrate_to_full_hits,
)

from callbacks import plotClusterSummary, plotEventDuringTraining
from argparse import ArgumentParser
from DeepJetCore.training.DeepJet_callbacks import simpleMetricsCallback


def EGNN_model(
    Inputs,
    td,
    debug_outdir=None,
    publishpath=None,
    plot_debug_every=record_frequency * plotfrequency,
):

    pre_selection = td.interpretAllModelInputs(Inputs, returndict=True)
    # tf.print("INPUTS SHAPE", {key: Inputs[key].shape for key in Inputs})
    # tf.print("PRE-SELECTION SHAPE", {key: Inputs[key].shape for key in pre_selection})
    pre_selection = condition_input(pre_selection, no_scaling=True, no_prime=True)
    print("available pre-selection outputs", list(pre_selection.keys()))

    rs = pre_selection["row_splits"]
    # is_track = pre_selection["is_track"]
    # x = ScaledGooeyBatchNorm2(fluidity_decay=0.01)(
    #    [pre_selection["features"], is_track]
    # )
    # x = ScaledGooeyBatchNorm2(fluidity_decay=0.01, invert_condition=True)([x, is_track])
    # x_in = Concatenate()([x, pre_selection["prime_coords"]])
    # x_in = Concatenate()([x_in, is_track, SphereActivation()(x_in)])
    # x_in = ScaledGooeyBatchNorm2(**BATCHNORM_OPTIONS)(x_in)
    h = pre_selection["no_coords"]
    coords = pre_selection["coords"]
    coords = coords / 3330
    h = Dense(128, activation=DENSE_ACTIVATION)(h)

    # x_in = Dense(64, activation=DENSE_ACTIVATION)(x_in)
    # x = x_in
    energy = pre_selection["rechit_energy"]
    # c_coords = pre_selection["prime_coords"]  # pre-clustered coordinates
    # c_coords = ScaledGooeyBatchNorm2(
    #     fluidity_decay=0.5,  # can freeze almost immediately
    # )(c_coords)
    t_idx = pre_selection["t_idx"]

    # c_coords = PlotCoordinates(
    #     plot_every=10, outdir=debug_outdir, name="input_c_coords", publish=publishpath
    # )([c_coords, energy, t_idx, rs])

    for i in range(0, 7):

        h, coords, gnnidx, gndist = RaggedEGCN(
            n_neighbours=11,
            n_dimensions=128,
            n_filters=128,
            n_propagate=128 * 2,
            coord_initialiser_noise=None,  # 1e-3,
            sumwnorm=True,
        )([h, rs], coords)

        # xgn = ScaledGooeyBatchNorm2(**BATCHNORM_OPTIONS)(xgn)

        # gndist = AverageDistanceRegularizer(strength=1e-2, record_metrics=True)(gndist)

        # gndist = LLRegulariseGravNetSpace(
        #     scale=0.01,
        #     print_loss=True,
        #     # project = True
        # )([gndist, pre_selection["prime_coords"], gnnidx])
        coords = PlotCoordinates(
            plot_every=plot_debug_every,
            outdir=debug_outdir,
            name="gn_coords_" + str(i),
            publish=publishpath,
        )([coords, energy, t_idx, rs])
        # coords = StopGradient()(coords)

    x = Concatenate()([h, coords])
    x = Dense(64, name="Last_Dense_1", activation=DENSE_ACTIVATION)(x)
    x = Dense(64, name="Last_Dense_2", activation=DENSE_ACTIVATION)(x)
    x = Dense(64, name="Last_Dense_3", activation=DENSE_ACTIVATION)(
        x
    )  # we want this to be not bounded

    ###########################################################################
    ########### the part below should remain almost unchanged #################
    ########### of course with the exception of the OC loss   #################
    ########### weights                                       #################
    ###########################################################################

    x = ScaledGooeyBatchNorm2(**BATCHNORM_OPTIONS)(x)
    # x = Concatenate()([x])
    (
        pred_beta,
        pred_ccoords,
        pred_dist,
        pred_energy_corr,
        pred_energy_low_quantile,
        pred_energy_high_quantile,
        pred_pos,
        pred_time,
        pred_time_unc,
        pred_id,
    ) = create_outputs(x, n_ccoords=3, fix_distance_scale=True)

    # pred_ccoords = LLFillSpace(maxhits=2000, runevery=5, scale=0.01)([pred_ccoords, rs, t_idx])

    # loss
    pred_beta = LLExtendedObjectCondensation(
        scale=1.0,
        use_energy_weights=False,  # well distributed anyways
        record_metrics=True,
        print_loss=True,
        name="ExtendedOCLoss",
        **LOSS_OPTIONS
    )(  # oc output and payload
        [
            pred_beta,
            pred_ccoords,
            pred_dist,
            pred_energy_corr,
            pred_energy_low_quantile,
            pred_energy_high_quantile,
            pred_pos,
            pred_time,
            pred_time_unc,
            pred_id,
        ]
        + [energy]
        +
        # truth information
        [
            pre_selection["t_idx"],
            pre_selection["t_energy"],
            pre_selection["t_pos"],
            pre_selection["t_time"],
            pre_selection["t_pid"],
            pre_selection["t_spectator_weight"],
            pre_selection["t_fully_contained"],
            pre_selection["t_rec_energy"],
            pre_selection["t_is_unique"],
            pre_selection["row_splits"],
        ]
    )

    # fast feedback
    pred_ccoords = PlotCoordinates(
        plot_every=plot_debug_every,
        outdir=debug_outdir,
        name="condensation",
        publish=publishpath,
    )([pred_ccoords, pred_beta, pre_selection["t_idx"], rs])
    model_outputs = {
        "pred_beta": pred_beta,
        "pred_ccoords": pred_ccoords,
        "pred_energy_corr_factor": pred_energy_corr,
        "pred_energy_low_quantile": pred_energy_low_quantile,
        "pred_energy_high_quantile": pred_energy_high_quantile,
        "pred_pos": pred_pos,
        "pred_time": pred_time,
        "pred_id": pred_id,
        "pred_dist": pred_dist,
        "rechit_energy": energy,
        "row_splits": pre_selection["row_splits"],
    }
    # #tf.print("MODEL OUTPUTS SHAPE", {k: v.shape for k, v in model_outputs.items()})
    return DictModel(inputs=Inputs, outputs=model_outputs)


import training_base_hgcal

train = training_base_hgcal.HGCalTraining()

# args = train.args
# learningrate = args.lr
# print("LR=", learningrate)

# if args.run_name != "":
#    wandb.init(project="hgcalml-1", tags=["debug", "small_dataset"], name=args.run_name)
#    wandb.run.log_code(".")
#    wandb.config["args"] = vars(args)
# nbatch = args.nbatch

PUBLISHPATH = "gkrzmanc@lxplus.cern.ch:/eos/home-g/gkrzmanc/results_aug23/"
PUBLISHPATH = None
if PUBLISHPATH is not None:
    PUBLISHPATH += [d for d in train.outputDir.split("/") if len(d)][-1]

publishpath = PUBLISHPATH  # this can be an ssh reachable path (be careful: needs tokens / keypairs)

if not train.modelSet():

    train.setModel(
        EGNN_model,
        td=train.train_data.dataclass(),
        publishpath=publishpath,
        debug_outdir=train.outputDir + "/intplots",
    )

    train.setCustomOptimizer(
        tf.keras.optimizers.Nadam(
            clipnorm=1.0,
            # epsilon=1e-2
        )
    )
    #
    train.compileModel(learningrate=1e-4)

    train.keras_model.summary()
    # wandb.watch(train.keras_model, log="all", log_freq=100)


verbosity = 2
import os

# establish callbacks

"""
simpleMetricsCallback(
    output_file=train.outputDir+'/metrics.html',
    record_frequency= record_frequency,
    plot_frequency = plotfrequency,
    select_metrics='FullOCLoss_*loss',
    publish=publishpath #no additional directory here (scp cannot create one)
    ),

simpleMetricsCallback(
    output_file=train.outputDir+'/latent_space_metrics.html',
    record_frequency= record_frequency,
    plot_frequency = plotfrequency,
    select_metrics='average_distance_*',
    publish=publishpath
    ),


simpleMetricsCallback(
    output_file=train.outputDir+'/val_metrics.html',
    call_on_epoch=True,
    select_metrics='val_*',
    publish=publishpath #no additional directory here (scp cannot create one)
    ),
"""


cb = [
    simpleMetricsCallback(
        output_file=train.outputDir + "/val_metrics.html",
        call_on_epoch=True,
        select_metrics="val_*",
        publish=publishpath,  # no additional directory here (scp cannot create one)
    ),
    simpleMetricsCallback(
        output_file=train.outputDir + "/metrics.html",
        record_frequency=record_frequency,
        plot_frequency=plotfrequency,
        select_metrics="*loss",
        publish=publishpath,  # no additional directory here (scp cannot create one)
    ),
]


# if args.run_name:
#    cb += [wandbCallback()]
#

num_events_to_plot = 10
if USE_WANDB:
    cb += [wandbCallback()]

plot_idx = []
for i in range(num_events_to_plot):
    plot_idx += [len(cb)]
    cb += [
        plotEventDuringTraining(
            outputfile=train.outputDir + "/cluster_coords/{}/".format(i),
            samplefile=train.train_data.getSamplePath(train.train_data.samples[0]),
            on_epoch_end=True,
            use_event=i,
        )
    ]

# FOR DEBUGGING
for i in plot_idx:
    cb[i].model = train.keras_model
    print("Calling callback", i, " (before training the model further)")
    cb[i].predict_and_call(1)


# FOR DEBUGGING
# for i in range(len(cb) - 1):
#    cb[i].model = train.keras_model
#    print("Calling callback", i, " (before training the model further)")
#    cb[i].predict_and_call(1)


train.change_learning_rate(LEARNINGRATE)

model, history = train.trainModel(nepochs=10, batchsize=NBATCH, additional_callbacks=cb)


print("freeze BN")
# Note the submodel here its not just train.keras_model
# for l in train.keras_model.layers:
#    if 'FullOCLoss' in l.name:
#        l.q_min/=2.

train.change_learning_rate(LEARNINGRATE / 5.0)

model, history = train.trainModel(
    nepochs=100, batchsize=NBATCH, additional_callbacks=cb
)
