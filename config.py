class Config:
    def __init__(self, args):
        self.core_res = args.core_res
        self.init_sparse_ratio = args.init_sparse_ratio
        self.iter_sparse_ratio = args.iter_sparse_ratio
        self.num_pruning_iters = args.num_pruning_iters

        if any([k in args.arch for k in ['bert-base-uncased', 'vit-base', 'm2f']]):
            self.hidden_dim = 768
        elif any([k in args.arch for k in ['bert-large-uncased', 'vit-large']]):
            self.hidden_dim = 1024

        global_flag = True

        self.training_params = {
            'model_default': {
                'data_default': {
                    'init': {
                        'num_train_epochs': 2,
                        'learning_rate': 1e-5
                    },
                    'iter': {
                        'num_train_epochs': 2,
                        'learning_rate': 1e-5
                    }
                }
            },
            'vit-base': {
                'data_default': {
                    'init': {
                        'num_train_epochs': 2,
                        'learning_rate': 1e-4
                    },
                    'iter': {
                        'num_train_epochs': 2,
                        'learning_rate': 1e-4
                    }
                }
            },
            'vit-large': {
                'data_default': {
                    'init': {
                        'num_train_epochs': 2,
                        'learning_rate': 1e-4
                    },
                    'iter': {
                        'num_train_epochs': 2,
                        'learning_rate': 1e-4
                    }
                }
            },
            'm2f': {
                'data_default': {
                    'init': {
                        'num_train_epochs': 2,
                        'learning_rate': 1e-4
                    },
                    'iter': {
                        'num_train_epochs': 2,
                        'learning_rate': 1e-4
                    }
                },
                'cityscapes': {
                    'init': {
                        'num_train_epochs': 2,
                        'learning_rate': 1e-4
                    },
                    'iter': {
                        'num_train_epochs': 2,
                        'learning_rate': 1e-4
                    }
                },
                'kitti': {
                    'init': {
                        'num_train_epochs': 2,
                        'learning_rate': 1e-4
                    },
                    'iter': {
                        'num_train_epochs': 2,
                        'learning_rate': 1e-4
                    }
                },
            },
            'bert-base-uncased': {
                'data_default': {
                    'init': {
                        'num_train_epochs': 2,
                        'learning_rate': 2e-5
                    },
                    'iter': {
                        'num_train_epochs': 2,
                        'learning_rate': 2e-5
                    }
                }
            },
            'bert-large-uncased': {
                'data_default': {
                    'init': {
                        'num_train_epochs': 2,
                        'learning_rate': 2e-5
                    },
                    'iter': {
                        'num_train_epochs': 2,
                        'learning_rate': 2e-5
                    }
                }
            }
        }
        self.pruning_params = {
            'model_default': {
                'data_default': {
                    'init': {
                        'training_steps': 10,  # taylor
                        'global_flag': global_flag,  # taylor
                        'num_iters': self.num_pruning_iters,  # perform taylor in x iters
                        'attn': {'sparse_ratio': self.init_sparse_ratio,
                                 'max_sparse_ratio': 0.85,
                                 'granularity': [self.core_res, self.hidden_dim]},
                        'ffn': {'sparse_ratio': self.init_sparse_ratio,
                                'max_sparse_ratio': 0.85,
                                'granularity': [1, self.hidden_dim]}
                    },
                    'iter': {
                        'training_steps': 10,  # taylor
                        'global_flag': global_flag,  # taylor
                        'num_iters': 1,  # perform taylor in x iters
                        'attn': {'sparse_ratio': self.iter_sparse_ratio,
                                 'granularity': [1, self.hidden_dim]},
                        'ffn': {'sparse_ratio': self.iter_sparse_ratio,
                                'granularity': [1, self.hidden_dim]}
                    }
                }
            }
        }

    def get_init_training_params(self, model_name, data_name):
        default_params = self.training_params.get(model_name, self.training_params['model_default'])['data_default']['init']
        data_params = self.training_params.get(model_name, self.training_params['model_default']).get(data_name, {'init': {}})['init']
        return default_params | data_params

    def get_iter_training_params(self, model_name, data_name):
        default_params = self.training_params.get(model_name, self.training_params['model_default'])['data_default']['iter']
        data_params = self.training_params.get(model_name, self.training_params['model_default']).get(data_name, {'iter': {}})['iter']
        return default_params | data_params

    def get_init_pruning_params(self, model_name, data_name):
        default_params = self.pruning_params.get(model_name, self.pruning_params['model_default'])['data_default']['init']
        data_params = self.pruning_params.get(model_name, self.pruning_params['model_default']).get(data_name, {'init': {}})['init']
        return default_params | data_params

    def get_iter_pruning_params(self, model_name, data_name):
        default_params = self.pruning_params.get(model_name, self.pruning_params['model_default'])['data_default']['iter']
        data_params = self.pruning_params.get(model_name, self.pruning_params['model_default']).get(data_name, {'iter': {}})['iter']
        return default_params | data_params
