import argparse
import datetime
import os


def modify_args(args):
    if args.device == 'gpu' and args.gpu_idx:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_idx

    args.datetime = format(str(datetime.datetime.now()))
    args.mask_finetune_flag = args.iter_sparse_ratio != 0

    if args.task == 'glue':
        args.metric_name = "matthews_correlation" if args.data == "cola" else "accuracy"
    elif args.task == 'img_class':
        args.metric_name = 'accuracy'
    elif args.task == 'img_seg':
        args.metric_name = 'accuracy'
    else:
        raise NotImplementedError

    if 'cifar' in args.data:
        args.final_eval_split = 'test'
    else:
        args.final_eval_split = 'val'

    return args


model_names = ['bert-base-uncased', 'bert-large-uncased', 'vit-base', 'vit-large', 'm2f']

arg_parser = argparse.ArgumentParser(description='Pruning main script')

exp_group = arg_parser.add_argument_group('exp', 'experiment setting')
exp_group.add_argument('--save_path', default='output', type=str, metavar='SAVE',
                       help='path to the experiment logging directory')
exp_group.add_argument('--evaluate_from', default=None, type=str, metavar='PATH',
                       help='path to saved checkpoint (default: none)')
exp_group.add_argument('--run_mode', default='train', type=str, choices=['train', 'evaluate'], help='Script mode')
exp_group.add_argument('--seed', default=0, type=int, help='random seed')
exp_group.add_argument('--gpu_idx', default=None, type=str, help='Index of available GPU')
exp_group.add_argument('--device', default='cuda', type=str, choices=['cpu', 'cuda', 'mps'], help='Device type for finetuning')
exp_group.add_argument('--comp_device', default='cuda', type=str, choices=['cpu', 'cuda', 'mps'], help='Device type for pruning/masking operations')

# compression related
comp_group = arg_parser.add_argument_group('comp', 'compression setting')
comp_group.add_argument('--num_pruning_rounds', '-num_pr', default=10, type=int)
comp_group.add_argument('--core_res', '-res', default=64, type=float, help='Sparsity resolution')
comp_group.add_argument('--init_sparse_ratio', '-init_sparse', default=0.5, type=float, help='Pruning sparsity')
comp_group.add_argument('--iter_sparse_ratio', '-iter_sparse', default=-0.75, type=float, help='Finetuning sparsity')
comp_group.add_argument('--num_pruning_iters', '-num_pi', default=4, type=int, help='Gradually prune in x iters')

# dataset related
data_group = arg_parser.add_argument_group('data', 'dataset setting')
data_group.add_argument('--task', metavar='D', default='glue', choices=['glue', 'qa', 'img_class', 'img_seg'], help='task to work on')
data_group.add_argument('--data', metavar='D', default='cola', help='data to work on')
data_group.add_argument('--data_root', metavar='DIR', default='data', help='path to dataset folder (default: data)')
data_group.add_argument('-j', '--workers', default=1, type=int, help='number of data loading workers (default: 1)')

# model arch related
arch_group = arg_parser.add_argument_group('arch', 'model architecture setting')
arch_group.add_argument('--arch', '-a', metavar='ARCH', default='bert-base-uncased',
                        type=str, choices=model_names,
                        help='model architecture: ' +
                             ' | '.join(model_names) +
                             ' (default: bert-base-uncased)')
