#! /bin/bash

MODEL="/eos/user/m/mgarciam/datasets_mlpf/models_trained/all_energies_10_15/hgcal/logs_1015_1911"
data_collection="/eos/user/m/mgarciam/datasets_mlpf/Evaluation_datasets/all_E_pandora/"
model_name="logs_1015_1911"
# mkdir "${data_collection}/pandora/"

# python3 scripts/predict_hgcal.py "${MODEL}/KERAS_check_best_model.h5" "${data_collection}/hgcal_fcc_eval/dataCollection.djcdc"  "${data_collection}/pandora/"

# mkdir "${data_collection}/pandora/analysis_${model_name}/"

python3 scripts/analyse_hgcal_predictions.py "${data_collection}/pandora/"  --analysisoutpath "${data_collection}/pandora/analysis_${model_name}/out.bin.gz"


