#!/bin/bash
#$ -N GatedGCN_ZINC_lspe-eigloss-ngape1-eq
#$ -q gpu
#$ -l gpu_card=1
#$ -t 1-4:1

pos_enc_dim=(0 4 6 8 16)
fname=$(pwd)/lspe-eigloss-ngape1-eq_${SGE_TASK_ID}_${pos_enc_dim[${SGE_TASK_ID}]}_DEBUG.log
touch $fname
fsync -d 10 $fname &

conda activate gnn
cd /afs/crc.nd.edu/user/p/psoga/benchmarking-gnns

python3 main_ZINC_graph_regression.py --config tests/test-configs/GatedGCN_ZINC_lspe-eigloss-ngape1-eq.json --job_num ${SGE_TASK_ID} --pos_enc_dim ${pos_enc_dim[${SGE_TASK_ID}]} --log_file $fname


# {'dataset': 'ZINC',
#  'gpu': {'id': 0, 'use': True},
#  'model': 'GatedGCN',
#  'net_params': {'L': 16,
#                 'alpha_loss': 1,
#                 'batch_norm': True,
#                 'batch_size': 128,
#                 'dropout': 0.0,
#                 'edge_feat': True,
#                 'gpu_id': 0,
#                 'hidden_dim': 59,
#                 'in_feat_dropout': 0.0,
#                 'lambda_loss': 0.1,
#                 'n_gape': 1,
#                 'out_dim': 59,
#                 'pe_init': 'gape',
#                 'pos_enc_dim': 20,
#                 'readout': 'mean',
#                 'residual': True,
#                 'use_lapeig_loss': True},
#  'out_dir': 'out/ZINC_graph_regression_lspe-eigloss-ngape1-eq',
#  'params': {'batch_size': 128,
#             'epochs': 1000,
#             'init_lr': 0.001,
#             'lr_reduce_factor': 0.5,
#             'lr_schedule_patience': 25,
#             'max_time': 12,
#             'min_lr': 1e-06,
#             'print_epoch_interval': 5,
#             'seed': 41,
#             'weight_decay': 0.0}}



# Generated with command:
#python3 configure_tests.py --config ../configs/GatedGCN_ZINC_LSPE_withLapEigLoss.json --pe_init gape --n_gape 1 --job_note lspe-eigloss-ngape1-eq --param_values 4 6 8 16
