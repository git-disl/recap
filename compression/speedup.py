import torch
from transformers.pytorch_utils import prune_linear_layer

from paths import get_path


def speedup(args, model, masks):
    if 'bert' in args.arch:
        return speedup_bert(args, model, masks)
    elif 'vit' in args.arch:
        return speedup_vit(args, model, masks)
    elif 'm2f' in args.arch:
        return speedup_swin_m2f(args, model, masks)
    else:
        raise NotImplementedError


def speedup_bert(args, model, masks):
    if isinstance(model, str):
        model = torch.load(model, map_location='cpu')

    if isinstance(masks, str):
        masks = torch.load(masks, map_location='cpu')

    def _prune_head_idxs(mask, num_heads):
        head_mask = (mask.reshape([num_heads, -1]).sum(-1) == 0.)
        return torch.arange(len(head_mask))[head_mask].long().tolist()

    # prune heads
    attention_modules = dict(
        [(name, module) for name, module in model.named_modules() if name.split('.')[-1] == 'attention'])
    for name, att_module in attention_modules.items():
        mask = masks[name + '.self.query']['weight'].to('cpu')
        num_heads = att_module.self.num_attention_heads
        prune_head_idxs = _prune_head_idxs(mask, num_heads)
        att_module.prune_heads(prune_head_idxs)
        att_module.pruned_heads = set()

    # prune ffns
    module_names = [name for name, _ in model.named_modules()]
    for name in module_names:
        if name not in masks.keys():
            continue
        if 'attention' not in name:
            module = model.get_submodule(name)
            if 'output' in name:
                module = prune_linear_layer(module, masks[name]['weight'].sum(dim=0).nonzero()[:, 0], dim=1)
            else:
                module = prune_linear_layer(module, masks[name]['weight'].sum(dim=1).nonzero()[:, 0])
            setattr(model.get_submodule('.'.join(name.split('.')[:-1])), name.split('.')[-1], module)

    return model


def speedup_vit(args, model, masks):
    if isinstance(model, str):
        model = torch.load(model, map_location='cpu')

    if isinstance(masks, str):
        masks = torch.load(masks, map_location='cpu')

    def _prune_head_idxs(mask, num_heads):
        head_mask = (mask.reshape([num_heads, -1]).sum(-1) == 0.)
        return torch.arange(len(head_mask))[head_mask].long().tolist()

    # prune heads
    attention_modules = dict(
        [(name, module) for name, module in model.named_modules() if name.split('.')[-1] == 'attention' and name.split('.')[-2] != 'attention'])
    for name, att_module in attention_modules.items():
        mask = masks[name + '.attention.query']['weight'].to('cpu')
        num_heads = att_module.attention.num_attention_heads
        prune_head_idxs = _prune_head_idxs(mask, num_heads)
        att_module.prune_heads(prune_head_idxs)
        att_module.pruned_heads = set()

    # prune ffns
    module_names = [name for name, _ in model.named_modules()]
    for name in module_names:
        if name not in masks.keys():
            continue
        if 'attention' not in name:
            module = model.get_submodule(name)
            if 'output' in name:
                module = prune_linear_layer(module, masks[name]['weight'].sum(dim=0).nonzero()[:, 0], dim=1)
            else:
                module = prune_linear_layer(module, masks[name]['weight'].sum(dim=1).nonzero()[:, 0])
            setattr(model.get_submodule('.'.join(name.split('.')[:-1])), name.split('.')[-1], module)

    return model


def speedup_swin_m2f(args, model, masks):
    if isinstance(model, str):
        model = torch.load(model, map_location='cpu')

    if isinstance(masks, str):
        masks = torch.load(masks, map_location='cpu')

    def _prune_head_idxs(mask, num_heads):
        head_mask = (mask.reshape([num_heads, -1]).sum(-1) == 0.)
        return torch.arange(len(head_mask))[head_mask].long().tolist()

    # prune heads
    # attention_modules = dict(
    #     [(name, module) for name, module in model.named_modules() if name.split('.')[-1] == 'attention'])

    attention_modules = dict(
        [(name, module) for name, module in model.named_modules() if name.split('.')[-1] == 'attention' and name.split('.')[-2] != 'attention'])
    for name, att_module in attention_modules.items():
        mask = masks[name + '.self.query']['weight'].to('cpu')
        num_heads = att_module.self.num_attention_heads
        prune_head_idxs = _prune_head_idxs(mask, num_heads)
        att_module.prune_heads(prune_head_idxs)
        rem_heads = [i for i in range(att_module.self.relative_position_bias_table.shape[-1]) if i not in prune_head_idxs]
        att_module.self.relative_position_bias_table = torch.nn.Parameter(att_module.self.relative_position_bias_table[:, rem_heads])
        att_module.pruned_heads = set()

        mask = torch.zeros(num_heads, dtype=bool)
        mask[rem_heads] = True
        masks[name +'.self'] = {'relative_position_bias_table': mask}

    # prune ffns
    module_names = [name for name, _ in model.named_modules()]
    for name in module_names:
        if name not in masks.keys():
            continue
        if 'attention' not in name:
            module = model.get_submodule(name)
            if 'output' in name:
                module = prune_linear_layer(module, masks[name]['weight'].sum(dim=0).nonzero()[:, 0], dim=1)
            else:
                module = prune_linear_layer(module, masks[name]['weight'].sum(dim=1).nonzero()[:, 0])
            setattr(model.get_submodule('.'.join(name.split('.')[:-1])), name.split('.')[-1], module)

    torch.save(masks, get_path(args, 'INIT_MASKS_PATH'))

    return model