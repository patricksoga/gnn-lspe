#!/bin/bash
#$ -N PNA_ZINC_lspe-ngape1
#$ -q gpu
#$ -l gpu_card=1
#$ -t 1-4:1

pos_enc_dim=(0 3 4 6 8)
fname=$(pwd)/lspe-ngape1_${SGE_TASK_ID}_${pos_enc_dim[${SGE_TASK_ID}]}_DEBUG.log
touch $fname
fsync -d 10 $fname &

conda activate gnn
cd /afs/crc.nd.edu/user/p/psoga/benchmarking-gnns

python3 main_ZINC_graph_regression.py --config tests/test-configs/PNA_ZINC_lspe-ngape1.json --job_num ${SGE_TASK_ID} --pos_enc_dim ${pos_enc_dim[${SGE_TASK_ID}]} --log_file $fname


# {'dataset': 'ZINC',
#  'gpu': {'id': 0, 'use': True},
#  'model': 'PNA',
#  'net_params': {'L': 16,
#                 'aggregators': 'mean max min std',
#                 'alpha_loss': 0.0001,
#                 'batch_norm': True,
#                 'batch_size': 128,
#                 'divide_input_first': True,
#                 'divide_input_last': True,
#                 'dropout': 0.0,
#                 'edge_dim': 40,
#                 'edge_feat': True,
#                 'gpu_id': 0,
#                 'graph_norm': True,
#                 'gru': False,
#                 'hidden_dim': 55,
#                 'in_feat_dropout': 0.0,
#                 'lambda_loss': 1000,
#                 'n_gape': 1,
#                 'out_dim': 55,
#                 'pe_init': 'gape',
#                 'pos_enc_dim': 16,
#                 'posttrans_layers': 1,
#                 'pretrans_layers': 1,
#                 'readout': 'sum',
#                 'residual': True,
#                 'scalers': 'identity amplification attenuation',
#                 'towers': 5,
#                 'use_lapeig_loss': False},
#  'out_dir': 'out/ZINC_graph_regression_lspe-ngape1',
#  'params': {'batch_size': 128,
#             'epochs': 1000,
#             'init_lr': 0.001,
#             'lr_reduce_factor': 0.5,
#             'lr_schedule_patience': 25,
#             'max_time': 48,
#             'min_lr': 1e-06,
#             'print_epoch_interval': 5,
#             'seed': 41,
#             'weight_decay': 3e-06}}



# Generated with command:
#python3 configure_tests.py --config ../configs/PNA_ZINC_LSPE.json --pe_init gape --n_gape 1 --job_note lspe-ngape1 --param_values 3 4 6 8
