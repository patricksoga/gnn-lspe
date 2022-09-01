#!/bin/bash
#$ -N GatedGCN_ZINC_gatedgcn_lspe_gapen3
#$ -q gpu
#$ -l gpu_card=1
#$ -t 1-6:1

pos_enc_dim=(0 6 8 12 16 20 64)
fname=$(pwd)/gatedgcn_lspe_gapen3_${SGE_TASK_ID}_${pos_enc_dim[${SGE_TASK_ID}]}_DEBUG.log
touch $fname
fsync -d 10 $fname &

conda activate gnn
cd /afs/crc.nd.edu/user/p/psoga/gnn-lspe

python3 main_ZINC_graph_regression.py --config tests/test-configs/GatedGCN_ZINC_gatedgcn_lspe_gapen3.json --job_num ${SGE_TASK_ID} --pos_enc_dim ${pos_enc_dim[${SGE_TASK_ID}]} --log_file $fname


# {'dataset': 'ZINC',
#  'gpu': {'id': 0, 'use': True},
#  'model': 'GatedGCN',
#  'net_params': {'L': 16,
#                 'alpha_loss': 0.0001,
#                 'batch_norm': True,
#                 'batch_size': 128,
#                 'dropout': 0.0,
#                 'edge_feat': True,
#                 'gape_pooling': 'sum',
#                 'gpu_id': 0,
#                 'hidden_dim': 59,
#                 'in_feat_dropout': 0.0,
#                 'lambda_loss': 1,
#                 'n_gape': 3,
#                 'out_dim': 59,
#                 'pe_init': 'gape',
#                 'pos_enc_dim': 20,
#                 'readout': 'mean',
#                 'residual': True,
#                 'use_lapeig_loss': False},
#  'out_dir': 'out/ZINC_graph_regression_gatedgcn_lspe_gapen3',
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
#python3 configure_tests.py --config ../configs/GatedGCN_ZINC_LSPE.json --pe_init gape --param_values 6 8 12 16 20 64 --job_note gatedgcn_lspe_gapen3 --n_gape 3 --gape_pooling sum
