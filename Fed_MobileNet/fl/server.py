import copy
import torch

def server_train(global_model, client_state_dicts, client_sizes):
    new_model = copy.deepcopy(global_model)
    new_state = new_model.state_dict()
    n_total = sum(client_sizes)

    for key in new_state.keys():
        new_state[key] = sum( (client_sizes[i] / n_total) * client_state_dicts[i][key] for i in range(len(client_state_dicts)) )

    new_model.load_state_dict(new_state)
    
    return new_model