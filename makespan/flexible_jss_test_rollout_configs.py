def same_param(param1, param2):
    if len(param1) != len(param2):
        return False
    
    for p1, p2 in zip(param1, param2):
        if abs(p1 - p2) > 1e-5:
            return False
    
    return True


######################### Get parameter search configurations ######################### 
def get_param_search_configs(num_oracle_trials, oracle_time_limit, oracle_stop_search_time):
    configurations = [
        ('default', {'action_type': 'default'}), 
        ('oracle', {'action_type': 'oracle', 'num_oracle_trials': num_oracle_trials,
            'oracle_time_limit': oracle_time_limit, 
            'oracle_stop_search_time': oracle_stop_search_time}), 
        ('warm_start_all', {'action_type': 'fix_all', 'do_warm_start': True}),
    ]

    for first_k in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8]:  
        configurations += [(f'first {first_k}', {'action_type': 'first', 'first_frac': first_k})]

    for random_p in [0.1, 0.2, 0.3, 0.4]: 
        configurations += [(f'random {random_p}', {'action_type': 'random', 'random_p': random_p})]

    return configurations


########################## Get Test Rollout Configurations ##########################
def get_simple_configs(oracle_params, model, model_th):
    # calibrate model based on default (maybe)
    configurations = [
        (f'default [{oracle_params}]', {'action_type': 'default',  'window': oracle_params[0], 'step': oracle_params[1],  
                                        'time_limit': oracle_params[2], 'stop_search_time': oracle_params[3]}),   
        (f'model [{oracle_params}]', {'action_type': 'model', 'model': model, 'model_th': model_th,
                    'window': oracle_params[0], 'step': oracle_params[1], 
                    'time_limit': oracle_params[2], 'stop_search_time': oracle_params[3]}),
        (f'model (include model time) [{oracle_params}]', 
            {'action_type': 'model', 'model': model,  'model_th': model_th, 'include_model_time': True, 
             'window': oracle_params[0], 'step': oracle_params[1], 'time_limit': oracle_params[2], 'stop_search_time': oracle_params[3]}),
    ]
    return configurations


def get_default_configs(model, model_th, time_limit, stop_search_time):
    configurations = [
                ('default', {'action_type': 'default'}), 
                ('model', {'action_type': 'model', 'model': model, 'model_th': model_th}),
                ('model (include model time)', {'time_limit': time_limit, 'action_type': 'model', 'model': model, 
                 'model_th': model_th, 'include_model_time': True, 'stop_search_time': stop_search_time}),
    ]

    for first_k in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8]:
        configurations += [(f'first {first_k}', {'action_type': 'first', 'first_frac': first_k})]

    for random_p in [0.1, 0.2, 0.3, 0.4]:
        configurations += [(f'random {random_p}', {'action_type': 'random', 'random_p': random_p})]

    configurations += [(f'warm_start_all', {'action_type': 'fix_all', 'do_warm_start': True})]
    return configurations

