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
        (f'default', {'action_type': 'default'}), 
        (f'oracle', {'action_type': 'oracle', 'num_oracle_trials': num_oracle_trials,
            'oracle_time_limit': oracle_time_limit, 
            'oracle_stop_search_time': oracle_stop_search_time}), 
        (f'warm_start_all', {'action_type': 'warm_start_all', 'do_warm_start': True}),
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


def get_oracle_configs(oracle_params, model, model_th, num_oracle_trials, oracle_time_limit, oracle_stop_search_time):
    configurations = [
        (f'default [{oracle_params}]', {'action_type': 'default',  'window': oracle_params[0], 'step': oracle_params[1],  
                                        'time_limit': oracle_params[2], 'stop_search_time': oracle_params[3]}),   
        # model
        (f'model [{oracle_params}]', {'action_type': 'model', 'model': model, 'model_th': model_th,
                    'window': oracle_params[0], 'step': oracle_params[1], 
                    'time_limit': oracle_params[2], 'stop_search_time': oracle_params[3]}),
        # model time is typically negligible
        (f'model (include model time) [{oracle_params}]', 
            {'action_type': 'model', 'model': model,  'model_th': model_th, 'include_model_time': True, 
            'window': oracle_params[0], 'step': oracle_params[1], 'time_limit': oracle_params[2], 'stop_search_time': oracle_params[3]}),
    ]
    
    for first_k in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8]:  
         configurations += [(f'first {first_k} [{oracle_params}]', 
                        {'action_type': 'first', 'first_frac': first_k,
                        'window': oracle_params[0], 'step': oracle_params[1], 
                        'time_limit': oracle_params[2], 'stop_search_time': oracle_params[3]})]

    for random_p in [0.1, 0.2, 0.3, 0.4]: 
        configurations += [(f'random {random_p} [{oracle_params}]', 
                          {'action_type': 'random', 'random_p': random_p,
                           'window': oracle_params[0], 'step': oracle_params[1], 
                           'time_limit': oracle_params[2], 'stop_search_time': oracle_params[3]})]

    configurations += [
        (f'warm_start_all [{oracle_params}]', {'action_type': 'fix_all', 'do_warm_start': True,
            'window': oracle_params[0], 'step': oracle_params[1], 
            'time_limit': oracle_params[2], 'stop_search_time': oracle_params[3]}),
        # oracle
        (f'oracle [{oracle_params}]', {'action_type': 'oracle', 'num_oracle_trials': num_oracle_trials,
            'oracle_time_limit': oracle_time_limit, 'oracle_stop_search_time': oracle_stop_search_time,
            'window': oracle_params[0], 'step': oracle_params[1], 
            'time_limit': oracle_params[2], 'stop_search_time': oracle_params[3]}),
    ]
    return configurations


def get_best_configs(best_params):
    default_params = best_params['default']
    warm_start_all_params = best_params['warm_start_all'] if 'warm_start_all' in best_params else best_params['default']
    oracle_params = best_params['oracle']
    configurations = [
            (f'default [{default_params}]', {'action_type': 'default', 
                                             'window': default_params[0], 'step': default_params[1], 
                                             'time_limit': default_params[2], 'stop_search_time': default_params[3]}),       
            (f'warm_start_all [{warm_start_all_params}]', {'action_type': 'fix_all', 'do_warm_start': True,
                                        'window': warm_start_all_params[0], 'step': warm_start_all_params[1], 
                                        'time_limit': warm_start_all_params[2], 'stop_search_time': warm_start_all_params[3]})
    ]

    for first_k in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8]: 
        if f'first {first_k}' in best_params:
            first_k_params = best_params[f'first {first_k}']
            if not same_param(first_k_params, oracle_params):
                configurations += [(f'first {first_k} [{first_k_params}]', {'action_type': 'first', 'first_frac': first_k,
                                    'window': first_k_params[0], 'step': first_k_params[1], 
                                    'time_limit': first_k_params[2], 'stop_search_time': first_k_params[3]})]

    for random_p in [0.1, 0.2, 0.3, 0.4]: 
        if f'random {random_p}' in best_params:
            random_p_params = best_params[f'random {random_p}']
            if not same_param(random_p_params, oracle_params):
                configurations += [(f'random {random_p} [{random_p_params}]', {'action_type': 'random', 'random_p': random_p, 
                                'window': random_p_params[0], 'step': random_p_params[1], 
                                'time_limit': random_p_params[2], 'stop_search_time': random_p_params[3]})]
    return configurations


def get_default_configs(window, model, model_th, time_limit, stop_search_time, 
                       num_oracle_trials, oracle_time_limit, oracle_stop_search_time):
    configurations = [
                (f'default', {'action_type': 'default'}), 
                (f'default [{window}-{window // 2}]', {'action_type': 'default', 'window': window, 'step': window // 2}),
                (f'model', {'action_type': 'model', 'model': model, 'model_th': model_th}),
                (f'model fix schedule', {'action_type': 'model', 'model': model, 'model_th': model_th, 'fix_option': "schedule"}),
                (f'model (include model time)', {'time_limit': time_limit, 'action_type': 'model', 'model': model, 
                'model_th': model_th, 'include_model_time': True, 'stop_search_time': stop_search_time}),
                # ('model_warm_start', {'time_limit': time_limit, 'action_type': 'model', 'model': model, 'model_th': model_th, 'do_warm_start': True}),
                (f'model_and_warm_start', {'time_limit': time_limit, 'action_type': 'model_and_warm_start', 
                'model': model, 'model_th': model_th, 'do_warm_start': True, 'stop_search_time': stop_search_time}),
    ]

    for first_k in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8]:
        configurations += [(f'first {first_k}', {'action_type': 'first', 'first_frac': first_k}),]

    configurations += [(f'warm_start_all', {'action_type': 'fix_all', 'do_warm_start': True}),
                (f'oracle', {'action_type': 'oracle', 'num_oracle_trials': num_oracle_trials,
                  'oracle_time_limit': oracle_time_limit, 'oracle_stop_search_time': oracle_stop_search_time}),
                (f'oracle_and_warm_start', {'action_type': 'oracle_and_warm_start', 
                  'do_warm_start': True, 'num_oracle_trials': num_oracle_trials,
                  'oracle_time_limit': oracle_time_limit, 'oracle_stop_search_time': oracle_stop_search_time})
            ]
    return configurations