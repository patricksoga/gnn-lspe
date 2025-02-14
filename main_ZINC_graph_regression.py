import logging

def get_logger(logfile=None):
    _logfile = logfile if logfile else './DEBUG.log'
    """Global logger for every logging"""
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        '%(asctime)s:%(filename)s:%(lineno)s - %(funcName)20s(): %(message)s')

    if not logger.handlers:
        debug_handler = logging.FileHandler(_logfile)
        debug_handler.setFormatter(formatter)
        debug_handler.setLevel(logging.DEBUG)
        logger.addHandler(debug_handler)

    return logger


"""
    IMPORTING LIBS
"""
import dgl

import numpy as np
import os
import socket
import time
import random
import glob
import argparse, json
import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim
from torch.utils.data import DataLoader

from tensorboardX import SummaryWriter
from tqdm import tqdm

import matplotlib
import matplotlib.pyplot as plt
from data.gape_preprocess import add_automaton_encodings, add_multiple_automaton_encodings


class DotDict(dict):
    def __init__(self, **kwds):
        self.update(kwds)
        self.__dict__ = self






"""
    IMPORTING CUSTOM MODULES/METHODS
"""
from nets.ZINC_graph_regression.load_net import gnn_model # import all GNNS
from data.data import LoadData # import dataset




"""
    GPU Setup
"""
def gpu_setup(use_gpu, gpu_id):
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)  

    if torch.cuda.is_available() and use_gpu:
        print('cuda available with GPU:',torch.cuda.get_device_name(0))
        device = torch.device("cuda")
    else:
        print('cuda not available')
        device = torch.device("cpu")
    return device








"""
    VIEWING MODEL CONFIG AND PARAMS
"""
def view_model_param(MODEL_NAME, net_params):
    model = gnn_model(MODEL_NAME, net_params)
    total_param = 0
    print("MODEL DETAILS:\n")
    #print(model)
    for param in model.parameters():
        # print(param.data.size())
        total_param += np.prod(list(param.data.size()))
    print('MODEL/Total parameters:', MODEL_NAME, total_param)
    return total_param


"""
    TRAINING CODE
"""

def train_val_pipeline(MODEL_NAME, dataset, params, net_params, dirs, logger):
    t0 = time.time()
    per_epoch_time = []
    model = gnn_model(MODEL_NAME, net_params)
    device = net_params['device']
    model = model.to(device)

        
    DATASET_NAME = dataset.name

    if net_params['pe_init'] == 'lap_pe':
        tt = time.time()
        logger.info("[!] -LapPE: Initializing graph positional encoding with Laplacian PE.")
        dataset._add_lap_positional_encodings(net_params['pos_enc_dim'])
        logger.info("[!] Time taken: ", time.time()-tt)
    elif net_params['pe_init'] == 'rand_walk':
        tt = time.time()
        logger.info("[!] -LSPE: Initializing graph positional encoding with rand walk features.")
        dataset._init_positional_encodings(net_params['pos_enc_dim'], net_params['pe_init'])
        logger.info("[!] Time taken: ", time.time()-tt)
        
        tt = time.time()
        logger.info("[!] -LSPE (For viz later): Adding lapeigvecs to key 'eigvec' for every graph.")
        dataset._add_eig_vecs(net_params['pos_enc_dim'])
        logger.info("[!] Time taken: ", time.time()-tt)
    elif net_params['pe_init'] == 'gape':
        logger.info(f"[!] Adding random automaton graph positional encoding ({net_params['pos_enc_dim']}).")
        logger.info(f"[!] Using matrix: {net_params['matrix_type']}")
        if net_params.get('n_gape', 1) > 1:
            logger.info(f"[!] Using {net_params.get('n_gape', 1)} random automata.")
            dataset = add_multiple_automaton_encodings(dataset, model.gape_pe_layer.pos_transitions, model.gape_pe_layer.pos_initials)
        else:
            dataset = add_automaton_encodings(dataset, model.gape_pe_layer.pos_transitions[0], model.gape_pe_layer.pos_initials[0], diag=False, matrix=net_params['matrix_type'])
            logger.info(f'Time PE:{time.time()-t0}')
        
    if MODEL_NAME in ['SAN', 'GraphiT']:
        if net_params['full_graph']:
            st = time.time()
            logger.info("[!] Adding full graph connectivity..")
            dataset._make_full_graph() if MODEL_NAME == 'SAN' else dataset._make_full_graph((net_params['p_steps'], net_params['gamma']))
            logger.info('Time taken to add full graph connectivity: ',time.time()-st)
    
    trainset, valset, testset = dataset.train, dataset.val, dataset.test
        
    root_log_dir, root_ckpt_dir, write_file_name, write_config_file, viz_dir = dirs
    
    # Write the network and optimization hyper-parameters in folder config/
    with open(write_config_file + '.txt', 'w') as f:
        f.write("""Dataset: {},\nModel: {}\n\nparams={}\n\nnet_params={}\n\n\nTotal Parameters: {}\n\n"""                .format(DATASET_NAME, MODEL_NAME, params, net_params, net_params['total_param']))
        
    log_dir = os.path.join(root_log_dir, "RUN_" + str(0))
    writer = SummaryWriter(log_dir=log_dir)

    # setting seeds
    random.seed(params['seed'])
    np.random.seed(params['seed'])
    torch.manual_seed(params['seed'])
    if device.type == 'cuda':
        torch.cuda.manual_seed(params['seed'])
        torch.cuda.manual_seed_all(params['seed'])
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    logger.info(f"Training Graphs: {len(trainset)}", )
    logger.info(f"Validation Graphs: {len(trainset)}", )
    logger.info(f"Test Graphs: {len(trainset)}", )

    optimizer = optim.Adam(model.parameters(), lr=params['init_lr'], weight_decay=params['weight_decay'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                     factor=params['lr_reduce_factor'],
                                                     patience=params['lr_schedule_patience'],
                                                     verbose=True)
    
    epoch_train_losses, epoch_val_losses = [], []
    epoch_train_MAEs, epoch_val_MAEs = [], [] 
    
    # import train functions for all GNNs
    from train.train_ZINC_graph_regression import train_epoch_sparse as train_epoch, evaluate_network_sparse as evaluate_network

    train_loader = DataLoader(trainset, num_workers=4, batch_size=params['batch_size'], shuffle=True, collate_fn=dataset.collate)
    val_loader = DataLoader(valset, num_workers=4, batch_size=params['batch_size'], shuffle=False, collate_fn=dataset.collate)
    test_loader = DataLoader(testset, num_workers=4, batch_size=params['batch_size'], shuffle=False, collate_fn=dataset.collate)
    
    # At any point you can hit Ctrl + C to break out of training early.
    try:
        for epoch in range(params['epochs']):

            start = time.time()

            epoch_train_loss, epoch_train_mae, optimizer = train_epoch(model, optimizer, device, train_loader, epoch)
                
            epoch_val_loss, epoch_val_mae, __ = evaluate_network(model, device, val_loader, epoch)
            epoch_test_loss, epoch_test_mae, __ = evaluate_network(model, device, test_loader, epoch)
            del __
            
            epoch_train_losses.append(epoch_train_loss)
            epoch_val_losses.append(epoch_val_loss)
            epoch_train_MAEs.append(epoch_train_mae)
            epoch_val_MAEs.append(epoch_val_mae)

            writer.add_scalar('train/_loss', epoch_train_loss, epoch)
            writer.add_scalar('val/_loss', epoch_val_loss, epoch)
            writer.add_scalar('train/_mae', epoch_train_mae, epoch)
            writer.add_scalar('val/_mae', epoch_val_mae, epoch)
            writer.add_scalar('test/_mae', epoch_test_mae, epoch)
            writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], epoch)

            # print the results
            logger.info(f"Epoch [{epoch}] Train Loss: {epoch_train_loss:.4f} | Train MAE: {epoch_train_mae:.4f} | Val Loss: {epoch_val_loss:.4f} | Val MAE: {epoch_val_mae:.4f} | Test MAE: {epoch_test_mae:.4f} | Time: {time.time()-start:.4f}")

            per_epoch_time.append(time.time()-start)

            # Saving checkpoint
            ckpt_dir = os.path.join(root_ckpt_dir, "RUN_")
            if not os.path.exists(ckpt_dir):
                os.makedirs(ckpt_dir)
            torch.save(model.state_dict(), '{}.pkl'.format(ckpt_dir + "/epoch_" + str(epoch)))

            files = glob.glob(ckpt_dir + '/*.pkl')
            for file in files:
                epoch_nb = file.split('_')[-1]
                epoch_nb = int(epoch_nb.split('.')[0])
                if epoch_nb < epoch-1:
                    os.remove(file)

            scheduler.step(epoch_val_loss)

            if optimizer.param_groups[0]['lr'] < params['min_lr']:
                logger.info("\n!! LR EQUAL TO MIN LR SET.")
                break
            
            # Stop training after params['max_time'] hours
            if time.time()-t0 > params['max_time']*3600:
                logger.info('-' * 89)
                logger.info("Max_time for training elapsed {:.2f} hours, so stopping".format(params['max_time']))
                break
                
    except KeyboardInterrupt:
        logger.info('-' * 89)
        logger.info('Exiting from training early because of KeyboardInterrupt')
    
    test_loss_lapeig, test_mae, g_outs_test = evaluate_network(model, device, test_loader, epoch)
    train_loss_lapeig, train_mae, g_outs_train = evaluate_network(model, device, train_loader, epoch)
    
    logger.info("Test MAE: {:.4f}".format(test_mae))
    logger.info("Train MAE: {:.4f}".format(train_mae))
    logger.info("Convergence Time (Epochs): {:.4f}".format(epoch))
    logger.info("TOTAL TIME TAKEN: {:.4f}s".format(time.time()-t0))
    logger.info("AVG TIME PER EPOCH: {:.4f}s".format(np.mean(per_epoch_time)))
    
    
    if net_params['pe_init'] in ('rand_walk', 'gape'):
        # Visualize actual and predicted/learned eigenvecs
        from utils.plot_util import plot_graph_eigvec
        if not os.path.exists(viz_dir):
            os.makedirs(viz_dir)

        sample_graph_ids = [15,25,45]

        for f_idx, graph_id in enumerate(sample_graph_ids):

            # Test graphs
            g_dgl = g_outs_test[graph_id]

            f = plt.figure(f_idx, figsize=(12,6))

            plt1 = f.add_subplot(121)
            plot_graph_eigvec(plt1, graph_id, g_dgl, feature_key='pos_enc', actual_eigvecs=True)

            plt2 = f.add_subplot(122)
            plot_graph_eigvec(plt2, graph_id, g_dgl, feature_key='p', predicted_eigvecs=True)

            f.savefig(viz_dir+'/test'+str(graph_id)+'.jpg')

            # Train graphs
            g_dgl = g_outs_train[graph_id]

            f = plt.figure(f_idx, figsize=(12,6))

            plt1 = f.add_subplot(121)
            plot_graph_eigvec(plt1, graph_id, g_dgl, feature_key='pos_enc', actual_eigvecs=True)

            plt2 = f.add_subplot(122)
            plot_graph_eigvec(plt2, graph_id, g_dgl, feature_key='p', predicted_eigvecs=True)

            f.savefig(viz_dir+'/train'+str(graph_id)+'.jpg')

    writer.close()

    """
        Write the results in out_dir/results folder
    """
    with open(write_file_name + '.txt', 'w') as f:
        f.write("""Dataset: {},\nModel: {}\n\nparams={}\n\nnet_params={}\n\n{}\n\nTotal Parameters: {}\n\n
    FINAL RESULTS\nTEST MAE: {:.4f}\nTRAIN MAE: {:.4f}\n\n
    Convergence Time (Epochs): {:.4f}\nTotal Time Taken: {:.4f} hrs\nAverage Time Per Epoch: {:.4f} s\n\n\n"""\
          .format(DATASET_NAME, MODEL_NAME, params, net_params, model, net_params['total_param'],
                  test_mae, train_mae, epoch, (time.time()-t0)/3600, np.mean(per_epoch_time)))
        




def main():    
    """
        USER CONTROLS
    """
    
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', help="Please give a config.json file with training/model/data/param details")
    parser.add_argument('--gpu_id', help="Please give a value for gpu id")
    parser.add_argument('--model', help="Please give a value for model name")
    parser.add_argument('--dataset', help="Please give a value for dataset name")
    parser.add_argument('--out_dir', help="Please give a value for out_dir")
    parser.add_argument('--seed', help="Please give a value for seed")
    parser.add_argument('--epochs', help="Please give a value for epochs")
    parser.add_argument('--batch_size', help="Please give a value for batch_size")
    parser.add_argument('--init_lr', help="Please give a value for init_lr")
    parser.add_argument('--lr_reduce_factor', help="Please give a value for lr_reduce_factor")
    parser.add_argument('--lr_schedule_patience', help="Please give a value for lr_schedule_patience")
    parser.add_argument('--min_lr', help="Please give a value for min_lr")
    parser.add_argument('--weight_decay', help="Please give a value for weight_decay")
    parser.add_argument('--print_epoch_interval', help="Please give a value for print_epoch_interval")    
    parser.add_argument('--L', help="Please give a value for L")
    parser.add_argument('--hidden_dim', help="Please give a value for hidden_dim")
    parser.add_argument('--out_dim', help="Please give a value for out_dim")
    parser.add_argument('--residual', help="Please give a value for residual")
    parser.add_argument('--edge_feat', help="Please give a value for edge_feat")
    parser.add_argument('--readout', help="Please give a value for readout")
    parser.add_argument('--in_feat_dropout', help="Please give a value for in_feat_dropout")
    parser.add_argument('--dropout', help="Please give a value for dropout")
    parser.add_argument('--layer_norm', help="Please give a value for layer_norm")
    parser.add_argument('--batch_norm', help="Please give a value for batch_norm")
    parser.add_argument('--max_time', help="Please give a value for max_time")
    parser.add_argument('--pos_enc_dim', help="Please give a value for pos_enc_dim")
    parser.add_argument('--pos_enc', help="Please give a value for pos_enc")
    parser.add_argument('--alpha_loss', help="Please give a value for alpha_loss")
    parser.add_argument('--lambda_loss', help="Please give a value for lambda_loss")
    parser.add_argument('--pe_init', help="Please give a value for pe_init")
    parser.add_argument('--n_gape', help="Please give a value for n_gape")
    parser.add_argument('--job_num')
    parser.add_argument('--log_file')
    args = parser.parse_args()
    with open(args.config) as f:
        config = json.load(f)
        
    # device
    if args.gpu_id is not None:
        config['gpu']['id'] = int(args.gpu_id)
        config['gpu']['use'] = True
    device = gpu_setup(config['gpu']['use'], config['gpu']['id'])
    # model, dataset, out_dir
    if args.model is not None:
        MODEL_NAME = args.model
    else:
        MODEL_NAME = config['model']
    if args.dataset is not None:
        DATASET_NAME = args.dataset
    else:
        DATASET_NAME = config['dataset']
    dataset = LoadData(DATASET_NAME)
    if args.out_dir is not None:
        out_dir = args.out_dir
    else:
        out_dir = config['out_dir']
    # parameters

    params = config['params']
    if args.seed is not None:
        params['seed'] = int(args.seed)
    if args.epochs is not None:
        params['epochs'] = int(args.epochs)
    if args.batch_size is not None:
        params['batch_size'] = int(args.batch_size)
    if args.init_lr is not None:
        params['init_lr'] = float(args.init_lr)
    if args.lr_reduce_factor is not None:
        params['lr_reduce_factor'] = float(args.lr_reduce_factor)
    if args.lr_schedule_patience is not None:
        params['lr_schedule_patience'] = int(args.lr_schedule_patience)
    if args.min_lr is not None:
        params['min_lr'] = float(args.min_lr)
    if args.weight_decay is not None:
        params['weight_decay'] = float(args.weight_decay)
    if args.print_epoch_interval is not None:
        params['print_epoch_interval'] = int(args.print_epoch_interval)
    if args.max_time is not None:
        params['max_time'] = float(args.max_time)
    # network parameters
    net_params = config['net_params']
    net_params['device'] = device
    net_params['gpu_id'] = config['gpu']['id']
    net_params['batch_size'] = params['batch_size']
    if args.L is not None:
        net_params['L'] = int(args.L)
    if args.hidden_dim is not None:
        net_params['hidden_dim'] = int(args.hidden_dim)
    if args.out_dim is not None:
        net_params['out_dim'] = int(args.out_dim)   
    if args.residual is not None:
        net_params['residual'] = True if args.residual=='True' else False
    if args.edge_feat is not None:
        net_params['edge_feat'] = True if args.edge_feat=='True' else False
    if args.readout is not None:
        net_params['readout'] = args.readout
    if args.in_feat_dropout is not None:
        net_params['in_feat_dropout'] = float(args.in_feat_dropout)
    if args.dropout is not None:
        net_params['dropout'] = float(args.dropout)
    if args.layer_norm is not None:
        net_params['layer_norm'] = True if args.layer_norm=='True' else False
    if args.batch_norm is not None:
        net_params['batch_norm'] = True if args.batch_norm=='True' else False
    if args.pos_enc is not None:
        net_params['pos_enc'] = True if args.pos_enc=='True' else False
    if args.pos_enc_dim is not None:
        net_params['pos_enc_dim'] = int(args.pos_enc_dim)
    if args.alpha_loss is not None:
        net_params['alpha_loss'] = float(args.alpha_loss)
    if args.lambda_loss is not None:
        net_params['lambda_loss'] = float(args.lambda_loss)
    if args.pe_init is not None:
        net_params['pe_init'] = args.pe_init
    if args.n_gape is not None:
        net_params['n_gape'] = int(args.n_gape)
    if args.log_file is not None:
        net_params['log_file'] = args.log_file
    
    # ZINC
    net_params['num_atom_type'] = dataset.num_atom_type
    net_params['num_bond_type'] = dataset.num_bond_type

    global logger
    logger = get_logger(net_params['log_file'])
    logger.info(params)
    logger.info(net_params)

    if MODEL_NAME == 'PNA':
        D = torch.cat([torch.sparse.sum(g.adjacency_matrix(transpose=True), dim=-1).to_dense() for g in
                       dataset.train.graph_lists])
        net_params['avg_d'] = dict(lin=torch.mean(D),
                                   exp=torch.mean(torch.exp(torch.div(1, D)) - 1),
                                   log=torch.mean(torch.log(D + 1)))
    
    root_log_dir = out_dir + 'logs/' + MODEL_NAME + "_" + DATASET_NAME + "_GPU" + str(config['gpu']['id']) + "_" + time.strftime('%Hh%Mm%Ss_on_%b_%d_%Y')
    root_ckpt_dir = out_dir + 'checkpoints/' + MODEL_NAME + "_" + DATASET_NAME + "_GPU" + str(config['gpu']['id']) + "_" + time.strftime('%Hh%Mm%Ss_on_%b_%d_%Y')
    write_file_name = out_dir + 'results/result_' + MODEL_NAME + "_" + DATASET_NAME + "_GPU" + str(config['gpu']['id']) + "_" + time.strftime('%Hh%Mm%Ss_on_%b_%d_%Y')
    write_config_file = out_dir + 'configs/config_' + MODEL_NAME + "_" + DATASET_NAME + "_GPU" + str(config['gpu']['id']) + "_" + time.strftime('%Hh%Mm%Ss_on_%b_%d_%Y')
    viz_dir = out_dir + 'viz/' + MODEL_NAME + "_" + DATASET_NAME + "_GPU" + str(config['gpu']['id']) + "_" + time.strftime('%Hh%Mm%Ss_on_%b_%d_%Y')
    dirs = root_log_dir, root_ckpt_dir, write_file_name, write_config_file, viz_dir

    if not os.path.exists(out_dir + 'results'):
        os.makedirs(out_dir + 'results')
        
    if not os.path.exists(out_dir + 'configs'):
        os.makedirs(out_dir + 'configs')

    net_params['total_param'] = view_model_param(MODEL_NAME, net_params)
    logger.info(f"Total number of parameters: {net_params['total_param']}")
    train_val_pipeline(MODEL_NAME, dataset, params, net_params, dirs, logger)

    
    
    
    
    
    
    
main()    





