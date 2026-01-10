import copy
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

def local_train(global_model, client_dataset, epochs=1, batch_size=64, lr=1e-3, device="cpu"):
    local_model = copy.deepcopy(global_model)
    local_model.to(device)
    local_model.train()
    loader = DataLoader(client_dataset, batch_size = batch_size, shuffle = True)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(local_model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)

    for _ in range(epochs):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            pred = local_model(x)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()

    n_k = len(client_dataset)
    state_dict_cpu = {}

    for k, v in local_model.state_dict().items():
        if torch.is_tensor(v):
            state_dict_cpu[k] = v.detach().cpu()
        else:
            state_dict_cpu[k] = v

    return state_dict_cpu, n_k

def local_train_prox(global_model, client_dataset, epochs=1, batch_size=64, lr=1e-3, device="cpu", mu=0.1):
    local_model = copy.deepcopy(global_model)
    local_model.to(device)
    global_params = [p.detach().to(device) for p in global_model.parameters()]
    local_model.train()
    loader = DataLoader(client_dataset, batch_size = batch_size, shuffle = True)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(local_model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)

    for _ in range(epochs):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            pred = local_model(x)
            loss = criterion(pred, y)

            # Add proximal term (mu/2) * ||w - w_global||^2
            if mu > 0:
                prox = 0.0
                for p, p0 in zip(local_model.parameters(), global_params):
                    prox += torch.sum((p - p0) ** 2)
                loss = loss + (mu / 2.0) * prox
                
            loss.backward()
            optimizer.step()

    n_k = len(client_dataset)
    state_dict_cpu = {}

    for k, v in local_model.state_dict().items():
        if torch.is_tensor(v):
            state_dict_cpu[k] = v.detach().cpu()
        else:
            state_dict_cpu[k] = v

    return state_dict_cpu, n_k