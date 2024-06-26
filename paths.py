def get_path(args, key: str, add_date=False, temp=True):

    method_tag = get_method_tag(args)

    if add_date:
        if method_tag:
            method_tag += '/'
        method_tag += args.datetime[:19]

    method_tag = method_tag.replace(' ', '_')

    folder_prefix = f'{args.save_path}/{args.task}/{args.data}/{args.arch}/{method_tag}'

    if temp:
        folder_prefix += '/temp'

    trainer_prefix = f'{folder_prefix}/trainer'
    model_prefix = f'{folder_prefix}/models'

    path_dict = {'MAIN_FOLDER_DIR': folder_prefix,
                 'TRAINER_FOLDER_DIR': trainer_prefix,
                 'MODEL_FOLDER_DIR': model_prefix,
                 'INIT_MODEL_PATH': f'{model_prefix}/init_model.pth',
                 'INIT_MASKS_PATH': f'{model_prefix}/init_masks.pth',
                 'ITER_MASKS_PATH': f'{model_prefix}/iter_masks.pth',
                 'OPT_STATE_PATH': f'{model_prefix}/opt_state.pth',
                 'COMPRESSED_MODEL_PATH': f'{model_prefix}/compressed_model.pth',
                 'ARGS_PATH': f'{folder_prefix}/args.json'
                 }

    return path_dict[key]


def get_method_tag(args):
    method_tag = []
    method_tag.append(str(args.init_sparse_ratio))
    method_tag.append(str(args.iter_sparse_ratio))

    method_tag = '_'.join(method_tag)
    if not method_tag:
        method_tag = '_'

    return method_tag
