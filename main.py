import gc
import json
import math
import shutil
import subprocess
from copy import deepcopy

import torch.optim
from torch.utils.data import ConcatDataset
from transformers import BertTokenizerFast, DataCollatorWithPadding, \
    ViTImageProcessor, Mask2FormerImageProcessor

import compression.pruner as compress_p
from args import arg_parser, modify_args
from config import *
from data_utils import prepare_datasets
from trainer_utils import *
from utils import get_model_param_keys

os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = 'true'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

args = arg_parser.parse_args()
args = modify_args(args)
torch.manual_seed(args.seed)

tokenizer_dispatcher = {
    'bert-base-uncased': BertTokenizerFast,
    'bert-large-uncased': BertTokenizerFast,
    'vit-base': ViTImageProcessor,
    'vit-large': ViTImageProcessor,
    'm2f': Mask2FormerImageProcessor
}


def finetune(model, args, data_content, training_params, model_path=None, for_eval_flag=True, tag='default'):
    trainer = prepare_traced_trainer(model, args, data_content, training_params, for_eval_flag=for_eval_flag, tag=tag)

    max_steps = math.ceil(training_params['num_train_epochs'] * len(data_content['train']))
    prepare_masked_trainer(args, trainer, max_steps)

    if os.path.exists(get_path(args, 'OPT_STATE_PATH')):
        opt_states = torch.load(get_path(args, 'OPT_STATE_PATH'))
        init_masks = torch.load(get_path(args, 'INIT_MASKS_PATH'))
        keys = get_model_param_keys(trainer.model)
        keys = keys[0] + keys[1]
        opt_states_to_load = trainer.optimizer.state_dict()

        for i in range(len(keys)):

            if 'embeddings.mask_token' in keys[i]:
                continue

            key_ = '.'.join(keys[i].split('.')[:-1])
            _key = keys[i].split('.')[-1]

            try:
                init_mask = init_masks[key_][_key].to('cpu')
            except:
                # print(f'Could not find init mask for {key}')
                init_mask = None

            if init_mask is not None:
                if _key == 'weight':
                    if ('attention' in key_ and ('query' in key_ or 'key' in key_ or 'value' in key_)) or \
                            ('intermediate' in key_):
                        init_mask = init_mask.sum(dim=1).nonzero()[:, 0]
                        opt_states_to_load['state'][i] = {
                            'step': opt_states[i]['step'],
                            'exp_avg': opt_states[i]['exp_avg'][init_mask].bfloat16(),
                            'exp_avg_sq': opt_states[i]['exp_avg_sq'][init_mask].bfloat16()}
                    elif 'output' in key_:
                        init_mask = init_mask.sum(dim=0).nonzero()[:, 0]
                        opt_states_to_load['state'][i] = {
                            'step': opt_states[i]['step'],
                            'exp_avg': opt_states[i]['exp_avg'][:, init_mask].bfloat16(),
                            'exp_avg_sq': opt_states[i]['exp_avg_sq'][:, init_mask].bfloat16()}
                    else:
                        raise NotImplementedError
                elif _key == 'relative_position_bias_table':
                    opt_states_to_load['state'][i] = {
                        'step': opt_states[i]['step'],
                        'exp_avg': opt_states[i]['exp_avg'][:, init_mask].bfloat16(),
                        'exp_avg_sq': opt_states[i]['exp_avg_sq'][:, init_mask].bfloat16()}
                else:
                    if ('attention' in key_ and ('query' in key_ or 'key' in key_ or 'value' in key_)) or \
                            ('intermediate' in key_):
                        init_mask = init_mask.nonzero()[:, 0]
                        opt_states_to_load['state'][i] = {
                            'step': opt_states[i]['step'],
                            'exp_avg': opt_states[i]['exp_avg'][init_mask].bfloat16(),
                            'exp_avg_sq': opt_states[i]['exp_avg_sq'][init_mask].bfloat16()}
                    elif 'output' in key_:
                        opt_states_to_load['state'][i] = {
                            'step': opt_states[i]['step'],
                            'exp_avg': opt_states[i]['exp_avg'].bfloat16(),
                            'exp_avg_sq': opt_states[i]['exp_avg_sq'].bfloat16()}
                    else:
                        raise NotImplementedError

        trainer.optimizer.load_state_dict(opt_states_to_load)

    trainer.train()

    trainer_state = trainer.state
    trainer_state.opt_state = trainer.optimizer.state_dict()['state']

    print('Completed finetuning')
    if model_path:
        torch.save(model, model_path)
        print(f'Saved to {model_path}')

    del trainer

    return model, trainer_state


def prepare_data(args, eval_key):
    if 'vit' in args.arch:
        tokenizer = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k", cache_dir='cache')
    elif 'm2f' in args.arch:
        tokenizer = Mask2FormerImageProcessor.from_pretrained("facebook/mask2former-swin-base-IN21k-cityscapes-semantic", cache_dir='cache')
    else:
        tokenizer = tokenizer_dispatcher[args.arch].from_pretrained(args.arch, cache_dir='cache')
    train_dataset, validation_datasets, test_dataset = prepare_datasets(args.arch, args.task, args.data, tokenizer,
                                                                        args.data_root, eval_key)

    dtype = torch.float32

    if args.task == 'img_class':
        def collate_fn_cls(examples):
            pixel_values = torch.stack([example["pixel_values"] for example in examples])
            if args.data == 'cifar100':
                labels = torch.tensor(np.array([example["fine_label"] for example in examples]))
            else:
                labels = torch.tensor(np.array([example["label"] for example in examples]))

            return {"pixel_values": pixel_values.to(dtype), "labels": labels}

        data_collator = collate_fn_cls
    elif args.task == 'img_seg':
        def collate_fn_seg(examples):
            data = []
            for key in examples[0].keys():
                if key == 'class_labels':
                    key_ = 'labels'
                else:
                    key_ = key

                if 'labels' in key:
                    val = [torch.tensor(np.stack(e[key], 0))[0] for e in examples]
                else:
                    val = np.concatenate([np.stack(e[key], 0) for e in examples])
                    val = torch.tensor(val).to(dtype)
                data.append((key_, val))
            return dict(data)

        data_collator = collate_fn_seg
    else:
        validation_datasets = ConcatDataset([d for d in validation_datasets.values()])
        data_collator = DataCollatorWithPadding(tokenizer)

    return {'train': train_dataset, 'val': validation_datasets, 'test': test_dataset,
            'collator': data_collator, 'tokenizer': tokenizer}


# @profile
def execute_main(args):
    model_name = args.arch

    if os.path.exists(get_path(args, 'MAIN_FOLDER_DIR', temp=False)):
        shutil.rmtree(get_path(args, 'MAIN_FOLDER_DIR', temp=False))
    Path(get_path(args, 'TRAINER_FOLDER_DIR')).mkdir(exist_ok=True, parents=True)
    Path(get_path(args, 'MODEL_FOLDER_DIR')).mkdir(exist_ok=True, parents=True)

    with open(get_path(args, 'ARGS_PATH'), "w") as f:
        json.dump(args.__dict__, f, indent=2)

    config = Config(args)
    data_content = prepare_data(args, 'val')

    if args.task == 'img_class':
        if args.data == 'cifar100':
            id2label = {id: label for id, label in enumerate(data_content['train'].features['fine_label'].names)}
        else:
            id2label = {id: label for id, label in enumerate(data_content['train'].features['label'].names)}

        label2id = {label: id for id, label in id2label.items()}
        model = build_model(model_name, args.task, args.data, id2label=id2label, label2id=label2id)
    else:
        model = build_model(model_name, args.task, args.data)

    torch.save(model, get_path(args, 'INIT_MODEL_PATH'))
    total_num_steps = 0

    print('init_prune_0 starts...')
    model = compress_p.init_pruning(model, args, config, data_content, tag='init_prune_0', beta=-1)
    if args.mask_finetune_flag:
        sparsity_ratio_mul = 1
        print('iter_prune_0 starts...')
        compress_p.iter_pruning(model, args, config, data_content, tag='iter_prune_0', sparsity_ratio_mul=sparsity_ratio_mul)
        model = torch.load(get_path(args, 'COMPRESSED_MODEL_PATH'), map_location=args.comp_device)
    else:
        model = model.to(args.comp_device)

    model_path = get_path(args, 'COMPRESSED_MODEL_PATH')

    print('finetune_0 starts')
    model = model.to(args.device)
    training_params = deepcopy(config.get_init_training_params(args.arch, args.data))

    _, trainer_state = finetune(model, args, data_content, training_params,
                                get_path(args, 'COMPRESSED_MODEL_PATH'), tag='finetune_0')
    total_num_steps += trainer_state.global_step

    Path(get_path(args, 'TRAINER_FOLDER_DIR', temp=False) + f'/runs').mkdir(exist_ok=True, parents=True)
    try:
        os.rename(get_path(args, 'TRAINER_FOLDER_DIR') + f'/runs/finetune_0',
                  get_path(args, 'TRAINER_FOLDER_DIR', temp=False) + f'/runs/finetune')
    except:
        pass

    tag = 'validate_0'
    print(f'{tag} starts')
    val_output = predict(model_path, args, data_content, tag=tag)
    val_score = val_output.metrics[f'{tag}_{args.metric_name}']

    best_val_score = val_score
    best_val_output = val_output
    subprocess.run(["cp", "-r", get_path(args, 'MODEL_FOLDER_DIR'), get_path(args, 'MAIN_FOLDER_DIR', temp=False)])

    num_rounds = args.num_pruning_rounds
    for i in range(num_rounds):
        print(f'Round: {i + 1}/{num_rounds} - Starting full model update...')
        init_model = compress_p.update_full_model(model, args, config, trainer_state, total_num_steps)
        print(f'Round: {i + 1}/{num_rounds} - Starting init pruning...')
        beta_ = -1
        model = compress_p.init_pruning(init_model, args, config, data_content,
                                        tag=f'init_prune_{i + 1}', beta=beta_)
        del init_model

        if args.mask_finetune_flag:
            sparsity_ratio_mul = i / max(1, num_rounds - 1)
            print(f'Round: {i + 1}/{num_rounds} - Starting iter pruning with mul: {sparsity_ratio_mul}')
            compress_p.iter_pruning(model, args, config, data_content,
                                    tag=f'iter_prune_{i + 1}',
                                    sparsity_ratio_mul=sparsity_ratio_mul)  # determine what to update
            model = torch.load(get_path(args, 'COMPRESSED_MODEL_PATH'), map_location=args.comp_device)

        training_params = deepcopy(config.get_iter_training_params(args.arch, args.data))

        print(f'Round: {i + 1}/{num_rounds} - Starting finetuning with initial learning rate '
              f'{training_params["learning_rate"]: .6f}')

        model = model.to(args.device)
        _, trainer_state = finetune(model, args, data_content, training_params,
                                    get_path(args, 'COMPRESSED_MODEL_PATH'),
                                    for_eval_flag=False, tag=f'finetune_{i + 1}')
        total_num_steps += trainer_state.global_step

        gc.collect()
        if args.device == 'mps':
            torch.mps.empty_cache()
        elif args.device == 'cuda':
            torch.cuda.empty_cache()
        gc.collect()

        print(f'Round: {i + 1}/{num_rounds} - Validating...')
        val_output = predict(model_path, args, data_content, tag=f'validate_{i + 1}')
        val_score = val_output.metrics[f'validate_{i + 1}_{args.metric_name}']

        if val_score >= best_val_score:
            best_val_score = val_score
            best_val_output = val_output
            subprocess.run(
                ["cp", "-r", get_path(args, 'MODEL_FOLDER_DIR'), get_path(args, 'MAIN_FOLDER_DIR', temp=False)])
            Path(get_path(args, 'TRAINER_FOLDER_DIR', temp=False) + f'/runs').mkdir(exist_ok=True, parents=True)

            subprocess.run(["rm", "-rf", get_path(args, 'TRAINER_FOLDER_DIR', temp=False) + f'/runs/finetune'])
            os.rename(get_path(args, 'TRAINER_FOLDER_DIR') + f'/runs/finetune_{i + 1}',
                      get_path(args, 'TRAINER_FOLDER_DIR', temp=False) + f'/runs/finetune')
        else:
            subprocess.run(["rm", "-rf", get_path(args, 'TRAINER_FOLDER_DIR') + f'/runs/finetune_{i + 1}'])

    print('Testing the finetuned model')
    model_path = get_path(args, 'COMPRESSED_MODEL_PATH', temp=False)
    test_output = predict(model_path, args, data_content, tag=args.final_eval_split)
    test_metric = test_output.metrics

    output_metric_dict = {'val_metric': best_val_output.metrics,
                          'test_metric': test_metric}

    subprocess.run(["rm", "-rf", get_path(args, 'MODEL_FOLDER_DIR')])

    return output_metric_dict


if __name__ == '__main__':

    run_mode = args.run_mode

    if run_mode == 'train':
        output_metric_dict = execute_main(args)
    elif run_mode == 'evaluate':
        model_path = args.evaluate_from
        data_content = prepare_data(args, args.final_eval_split)
        output_metric_dict = predict(model_path, args, data_content, tag='test')
    else:
        raise NotImplementedError
