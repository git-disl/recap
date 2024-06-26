import logging
from copy import deepcopy

from compression.speedup import speedup
from config_helpers import *
from general_utils import LogLevel
from nni.contrib.compression.pruning import TaylorPruner
from nni.contrib.compression.utils import TransformersEvaluator
from trainer_utils import *
from utils import get_model_param_keys

pruner_dispatcher = {'taylor': TaylorPruner}


def update_full_model(model, args, config, trainer_state, total_num_steps):
    init_model = torch.load(get_path(args, 'INIT_MODEL_PATH'), map_location='cpu')
    init_masks = torch.load(get_path(args, 'INIT_MASKS_PATH'), map_location='cpu')
    opt_found_flag = False
    keys = get_model_param_keys(model)
    keys = keys[0] + keys[1]

    if os.path.exists(get_path(args, 'OPT_STATE_PATH')):
        opt_states = torch.load(get_path(args, 'OPT_STATE_PATH'), map_location='cpu')
        opt_found_flag = True
    else:
        opt_states = dict([(i, {'step': 0}) for i in range(len(trainer_state.opt_state))])
    model = model.to('cpu')

    if args.mask_finetune_flag:
        iter_masks = torch.load(get_path(args, 'ITER_MASKS_PATH'), map_location='cpu')
    else:
        iter_masks = None

    init_model_state_dict = init_model.state_dict()
    model_state_dict = model.state_dict()

    for key, val in model_state_dict.items():
        key_ = '.'.join(key.split('.')[:-1])
        _key = key.split('.')[-1]

        if key not in keys:
            continue

        opt_idx = keys.index(key)

        if 'embeddings.mask_token' in key:
            continue

        try:
            init_mask = init_masks[key_][_key]
        except:
            # print(f'Could not find init mask for {key}')
            init_mask = None

        if 'relative_position_bias_table' in key:
            init_mask = init_mask.repeat([model_state_dict[key].shape[0], 1])

        try:
            iter_mask = iter_masks[key_][_key]  # update these values
        except:
            # print(f'Could not find iter mask for {key}')
            iter_mask = torch.ones_like(model_state_dict[key]).bool()

        if init_mask is None:  # check this
            if init_model_state_dict[key].shape != model_state_dict[key].shape:
                print(key)
                raise RuntimeError
            init_model_state_dict[key] = model_state_dict[key]

            if opt_found_flag:
                opt_states[opt_idx]['exp_avg'] = trainer_state.opt_state[opt_idx]['exp_avg'].to('cpu')
                opt_states[opt_idx]['exp_avg_sq'] = trainer_state.opt_state[opt_idx]['exp_avg_sq'].to('cpu')

        else:
            pad_idx = init_mask.flatten().nonzero().squeeze()[iter_mask.flatten() == 1]
            mask_padded = torch.zeros_like(init_mask).flatten()
            mask_padded[pad_idx] = 1
            mask_padded = mask_padded.reshape(init_mask.shape)

            try:
                init_model_state_dict[key][mask_padded] = model_state_dict[key][iter_mask].flatten()
            except:
                # print(f'Could not find update {key}')
                pass

            if opt_found_flag:
                try:
                    opt_states[opt_idx]['exp_avg'][mask_padded] = trainer_state.opt_state[opt_idx]['exp_avg'].to('cpu')[iter_mask].flatten()
                    opt_states[opt_idx]['exp_avg_sq'][mask_padded] = trainer_state.opt_state[opt_idx]['exp_avg_sq'].to('cpu')[iter_mask].flatten()
                    opt_states[opt_idx]['exp_avg'][~mask_padded] *= 0.9
                    opt_states[opt_idx]['exp_avg_sq'][~mask_padded] *= 0.999
                except:
                    print(key)

        opt_states[opt_idx]['step'] = int(trainer_state.opt_state[opt_idx]['step'].item() + opt_states[opt_idx]['step'])
        if not opt_found_flag:
            opt_states[opt_idx]['exp_avg'] = torch.zeros_like(init_model_state_dict[key])
            opt_states[opt_idx]['exp_avg_sq'] = torch.zeros_like(init_model_state_dict[key])

    init_model.load_state_dict(init_model_state_dict)
    torch.save(init_model, get_path(args, 'INIT_MODEL_PATH'))  # save the updated model
    torch.save(opt_states, get_path(args, 'OPT_STATE_PATH'))  # save the updated model

    return init_model


def init_pruning(model, args, config, data_content, tag='default', method=None, beta=-1):
    training_params = config.get_init_training_params(args.arch, args.data)
    pruning_params = config.get_init_pruning_params(args.arch, args.data)
    pruning_params['beta'] = beta

    full_masks = None
    cur_pruning_params = deepcopy(pruning_params)
    num_iters = pruning_params.get('num_iters', 1)
    for iter_idx in range(num_iters):
        cur_pruning_params['attn']['sparse_ratio'] = pruning_params['attn']['sparse_ratio'] \
                                                     / num_iters / (1 - iter_idx * pruning_params['attn']['sparse_ratio'] / num_iters)
        cur_pruning_params['ffn']['sparse_ratio'] = pruning_params['ffn']['sparse_ratio'] \
                                                    / num_iters / (1 - iter_idx * pruning_params['ffn']['sparse_ratio'] / num_iters)

        config_list = get_prune_config_for_attn(args, model, cur_pruning_params['attn']) \
                      + get_prune_config_for_ffn(args, model, cur_pruning_params['ffn'])

        for c in config_list:
            if not cur_pruning_params['global_flag']:
                del c['global_group_id']

        if method is None:
            method = 'taylor'

        if model.device != args.comp_device:
            model = model.to(args.comp_device)

        model, masks, _ = prune(model, args, data_content, training_params, cur_pruning_params,
                                config_list, method, tag=tag, device=args.comp_device)

        if full_masks is None:
            full_masks = deepcopy(masks)
        else:
            for k in masks.keys():
                for k_ in masks[k].keys():
                    if full_masks[k][k_] is None:
                        continue
                    pad_idx = full_masks[k][k_].flatten().nonzero().squeeze()[masks[k][k_].flatten() == 1]
                    mask_padded = torch.zeros_like(full_masks[k][k_]).flatten()
                    mask_padded[pad_idx] = 1
                    mask_padded = mask_padded.reshape(full_masks[k][k_].shape)
                    full_masks[k][k_][mask_padded == 0] = False

        if iter_idx == num_iters - 1:
            torch.save(model, get_path(args, 'COMPRESSED_MODEL_PATH'))
            torch.save(full_masks, get_path(args, 'INIT_MASKS_PATH'))

    return model


def iter_pruning(model, args, config, data_content, tag='default', method=None, sparsity_ratio_mul=0):
    training_params = config.get_iter_training_params(args.arch, args.data)
    pruning_params = config.get_iter_pruning_params(args.arch, args.data)
    init_pruning_params = config.get_init_pruning_params(args.arch, args.data)

    pruning_params['beta'] = 1
    cur_pruning_params = deepcopy(pruning_params)

    if sparsity_ratio_mul == 0:
        if cur_pruning_params['attn']['sparse_ratio'] < 0:
            cur_pruning_params['attn']['sparse_ratio'] *= -1
        if cur_pruning_params['ffn']['sparse_ratio'] < 0:
            cur_pruning_params['ffn']['sparse_ratio'] *= -1
    else:
        if cur_pruning_params['attn']['sparse_ratio'] < 0:
            cur_pruning_params['attn']['sparse_ratio'] *= -1
        else:
            cur_pruning_params['attn']['sparse_ratio'] += \
                (init_pruning_params['attn']['sparse_ratio'] -
                 pruning_params['attn']['sparse_ratio']) * sparsity_ratio_mul
        if cur_pruning_params['ffn']['sparse_ratio'] < 0:
            cur_pruning_params['ffn']['sparse_ratio'] *= -1
        else:
            cur_pruning_params['ffn']['sparse_ratio'] += \
                (init_pruning_params['ffn']['sparse_ratio'] -
                 pruning_params['ffn']['sparse_ratio']) * sparsity_ratio_mul

    config_list = get_prune_config_for_attn(args, model, cur_pruning_params['attn']) \
                  + get_prune_config_for_ffn(args, model, cur_pruning_params['ffn'])

    for c in config_list:
        if 'dependency_group_id' in c.keys():
            del c['dependency_group_id']
        if not cur_pruning_params['global_flag']:
            del c['global_group_id']

    if method is None:
        method = 'taylor'

    if model.device != args.comp_device:
        model = model.to(args.comp_device)

    model, masks, pruner = prune(model, args, data_content, training_params, cur_pruning_params,
                                 config_list, method, tag=tag, device=args.comp_device, speedup_flag=False)

    torch.save(masks, get_path(args, 'ITER_MASKS_PATH'))

    return model


# @profile
def prune(model, args, data_content, training_params, pruning_params, config_list, pruner_method,
          tag='default', device='cpu', speedup_flag=True):
    training_params = deepcopy(training_params)
    training_params['learning_rate'] = 0
    trainer = prepare_traced_trainer(model, args, data_content, training_params, for_train_flag=False,
                                     for_eval_flag=False, tag=tag, device=device, send_tag='train')
    evaluator = TransformersEvaluator(trainer)

    pruner_init_kwargs = {}
    pruner_compress_kwargs = {}
    if pruner_method == 'movement':
        pruner_init_kwargs = {'warmup_step': pruning_params['warmup_step'],
                              'cooldown_begin_step': pruning_params['cooldown_begin_step']}
        pruner_compress_kwargs = {'max_steps': pruning_params['cooldown_begin_step'],
                                  'max_epochs': training_params.get('num_train_epochs', 3)}
    elif pruner_method == 'taylor':
        pruner_init_kwargs = {'training_steps': pruning_params['training_steps'],
                              'beta': pruning_params['beta'],
                              'global_flag': pruning_params['global_flag']}

    with LogLevel(logging.ERROR):
        pruner = pruner_dispatcher[pruner_method](model, config_list, evaluator, **pruner_init_kwargs)
        pruner.compress(**pruner_compress_kwargs)
        pruner.unwrap_model()

    masks = pruner.get_masks()

    if speedup_flag:
        pruned_model = speedup(args, model.to('cpu'), masks)
    else:
        pruned_model = None

    return pruned_model, masks, pruner
