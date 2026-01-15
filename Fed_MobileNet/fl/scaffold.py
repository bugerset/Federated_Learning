import copy
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

def scaffold(global_model, client_dataset, c, c_i, epochs=1, batch_size=64, lr=1e-3, device="cpu"):
    local_model = copy.deepcopy(global_model)
    local_model.to(device)

    # Prepare control variates on device
    c_t = {k: v.to(device) for k, v in c.items()}
    c_i_t = {k: v.to(device) for k, v in c_i.items()}

    local_model.train()
    loader = DataLoader(client_dataset, batch_size=batch_size, shuffle=True)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(local_model.parameters(), lr=lr)
    steps = 0

    for _ in range(epochs):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            pred = local_model(x)
            loss = criterion(pred, y)
            loss.backward()
            
            with torch.no_grad():
                for name, param in local_model.named_parameters():
                    if param.grad is None:
                        continue
                    param.grad.add_(-c_i_t[name] + c_t[name])

            optimizer.step()
            steps += 1

    n_k = len(client_dataset)

    state_dict_cpu = {}
    for k, v in local_model.state_dict().items():
        if torch.is_tensor(v):
            state_dict_cpu[k] = v.detach().cpu()
        else:
            state_dict_cpu[k] = v

    global_params_cpu = {name: p.detach().cpu() for name, p in global_model.named_parameters()}
    local_params_cpu = {name: p.detach().cpu() for name, p in local_model.named_parameters()}

    if steps == 0:
        c_i_new = {k: v.clone() for k, v in c_i.items()}
        delta_c = {k: torch.zeros_like(v) for k, v in c_i.items()}
    else:
        c_i_new = {}
        delta_c = {}
        scale = 1.0 / (steps * lr)
        for name, c_val in c_i.items():
            update = c_val - c[name] + scale * (global_params_cpu[name] - local_params_cpu[name])
            c_i_new[name] = update
            delta_c[name] = update - c_val

    return state_dict_cpu, n_k, delta_c, c_i_new

def init_control_variates(global_model):
    control = {}
    for name, param in global_model.named_parameters():
        control[name] = torch.zeros_like(param.detach().cpu())
    return control

def update_server_c(server_c, delta_cs):
    if not delta_cs:
        return server_c

    m = len(delta_cs)
    new_c = {}
    
    for name, c_val in server_c.items():
        sum_delta = sum(dc[name] for dc in delta_cs)
        new_c[name] = c_val + sum_delta / m
    return new_c
