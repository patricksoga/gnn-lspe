import json
import os
import argparse
import torch
import pprint
import sys

import sys 
sys.path.append('..')

from util import add_args, get_net_params, get_parameters


def dataset_to_graph_task(dataset):
    if dataset == "ZINC":
        return "graph_regression"
    else:
        return "graph_classification"


def gpu_setup(use_gpu, gpu_id):
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)  

    if torch.cuda.is_available() and use_gpu:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    return device


def script_boilerplate(args):
    """
    Generates the string setting up the job information for the experiment.
    """
    num_cards = len(args.param_values)
    model = args.model
    dataset = args.dataset
    job_note = args.job_note
    return f"""#!/bin/bash
#$ -N {model}_{dataset}_{job_note}
#$ -q gpu
#$ -l gpu_card=1
#$ -t 1-{num_cards}:1

"""

def pre_run_boilerplate(args):
    """
    Generates the string setting up the environment for the experiment.
    """
    debug_file = f"fname=$(pwd)/{args.job_note}_${{SGE_TASK_ID}}_${{{args.varying_param}[${{SGE_TASK_ID}}]}}_DEBUG.log"
    rest = f"""touch $fname
fsync -d 10 $fname &

conda activate gnn
cd /afs/crc.nd.edu/user/p/psoga/benchmarking-gnns

"""
    return debug_file + "\n" + rest


def config_string(config):
    """
    Generates the string detailing parameters for the experiment.
    """
    pretty_params = pprint.pformat(config)

    pretty_params = '\n'.join('# ' + s for s in pretty_params.split('\n')) 

    return pretty_params + "\n"


def run_string(args, config_path):
    """
    Generates the string for running the experiment.
    """
    dataset = args.dataset
    graph_task = dataset_to_graph_task(dataset)
    return f"python3 main_{dataset}_{graph_task}.py --config {config_path} --job_num ${{SGE_TASK_ID}} --{args.varying_param} ${{{args.varying_param}[${{SGE_TASK_ID}}]}} --log_file $fname\n"


def main(args):
    if args.varying_param is None:
        raise ValueError('Must specify varying parameter')
    if args.varying_param and args.param_values is None:
        raise ValueError('Must specify param values')
    if args.job_note is None:
        raise ValueError('Must specify job note')

    script_string = ""

    with open(args.config) as f:
        config = json.load(f)
    params = get_parameters(config, args)
    device = gpu_setup(config['gpu']['use'], config['gpu']['id'])

    args.model = config["model"]
    if args.dataset is None:
        args.dataset = config["dataset"]
    else:
        config["dataset"] = args.dataset

    if args.dataset is not None:
        DATASET_NAME = args.dataset

    net_params = get_net_params(config, args, device, params, DATASET_NAME)

    model = args.model
    dataset = args.dataset

    config = {
        "gpu": {
            "use": config["gpu"]["use"],
            "id": config["gpu"]["id"]
        },
        "model": config["model"],
        "dataset": config["dataset"],
        "out_dir": f"out/{dataset}_{dataset_to_graph_task(dataset)}_{args.job_note}",
        "params": params,
        "net_params": net_params,
    }

    del config["net_params"]["device"]

    config_filename = f"./test-configs/{model}_{dataset}_{args.job_note}.json"
    with open(config_filename, "w+") as f:
        json.dump(config, f)

    config_filename = os.path.join('tests', '/'.join(config_filename.split('/')[1:]))

    script_string += script_boilerplate(args)

    varying_param_str = f"{args.varying_param}=({' '.join(['0'] + args.param_values)})"
    script_string += varying_param_str + "\n"
    script_string += pre_run_boilerplate(args)
    script_string += run_string(args, config_filename) + "\n\n"
    script_string += config_string(config) + "\n"

    generating_command = "python3 " + ' '.join(sys.argv)
    script_string += "\n\n# Generated with command:\n#" + generating_command + "\n"

    try:
        out_dir = f"../{config['out_dir']}"
        if not os.path.isdir(out_dir):
            os.makedirs(out_dir)

        exp_dataset_dir = f"./{dataset}" 
        if not os.path.isdir(exp_dataset_dir):
            os.makedirs(exp_dataset_dir)

        model_dataset_dir = f"{exp_dataset_dir}/{model}"
        if not os.path.isdir(model_dataset_dir):
            os.makedirs(model_dataset_dir)

        script_folder_1 = f"{model_dataset_dir}/{net_params['pe_init']}"
        if not os.path.isdir(script_folder_1):
            os.makedirs(script_folder_1)

        script_folder_2 = f"{script_folder_1}/{args.job_note}"
        if not os.path.isdir(script_folder_2):
            os.makedirs(script_folder_2)

        script_path = f"{script_folder_2}/{args.job_note}.sh"
        with open(script_path, 'w') as f:
            f.write(script_string)

        print("Script written to: ", script_path)

    except Exception as e:
        print(f"Could not write script to {script_path}")
        print(e)
        raise e


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--varying_param", type=str, help="Parameter to vary, only one allowed", default="pos_enc_dim")
    parser.add_argument("--param_values", nargs='+', help="Values to vary")
    parser.add_argument("--job_note", type=str, help="Job note for job name and script header")
    parser = add_args(parser)
    main(parser.parse_args())