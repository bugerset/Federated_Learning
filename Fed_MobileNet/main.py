import numpy as np
import copy
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from data import cifar10
from data.partition import IID_partition, NIID_partition, print_label_counts
from fl.fedavg import fedavg
from fl.fedprox import fedprox
from fl.scaffold import scaffold, init_control_variates, update_server_c
from fl.server import server_train
from models.mobilenet import MobileNet
from utils.seed import set_seed
from utils.eval import eval
from utils.parser import parse_args
from utils.device import select_device

def main():
    args = parse_args()
    set_seed(args.seed, True)
    rng = np.random.default_rng(args.seed)

    device = select_device(args.device)
    print(f"Device => {device}")

    train_ds, test_ds = cifar10.get_cifar10(root=args.data_root, normalize=args.normalize, augment=args.augment)
    test_loader = DataLoader(test_ds, batch_size=args.test_batch_size, shuffle=False)
    global_model = MobileNet(num_classes=10).to(device)

    lr_holder = nn.Parameter(torch.zeros(1, device=device), requires_grad=True)
    lr_opt = torch.optim.SGD([lr_holder], lr=args.lr)

    scheduler = ReduceLROnPlateau(
        lr_opt,
        mode="min",
        factor=args.lr_factor,
        patience=args.lr_patience,
        threshold=args.lr_threshold,
        cooldown=args.lr_cooldown,
        min_lr=args.min_lr,
    )

    current_lr = lr_opt.param_groups[0]["lr"]

    if args.train == "scaffold":
        server_c = init_control_variates(global_model)
        client_cs = [copy.deepcopy(server_c) for _ in range(args.num_clients)]
    else:
        server_c = None
        client_cs = None

    if args.partition == "niid":
        clients = NIID_partition(train_ds, num_clients=args.num_clients, seed=args.seed, alpha=args.alpha, min_size=args.min_size)
    else:
        clients = IID_partition(train_ds, num_clients=args.num_clients, seed=args.seed)

    if args.print_labels:
        print("\n=== Client label distributions ===")
        print_label_counts(train_ds, clients, num_classes=10)

    m = max(1, int(args.client_frac * args.num_clients))

    for r in range(args.rounds):
        selected = rng.choice(args.num_clients, size=m, replace=False).tolist()

        client_states = []
        client_sizes = []
        delta_cs = []

        for cid in selected:
            if args.train == "fedavg":
                state, n_k = fedavg(
                    global_model,
                    clients[cid],
                    epochs=args.local_epochs,
                    batch_size=args.batch_size,
                    lr=current_lr,
                    device=device
                )
            elif args.train == "fedsgd":
                state, n_k = fedavg(
                    global_model,
                    clients[cid],
                    epochs=1,
                    batch_size=len(clients[cid]),
                    lr=current_lr,
                    device=device
                )
            elif args.train == "fedprox":
                state, n_k = fedprox(
                    global_model,
                    clients[cid],
                    epochs=args.local_epochs,
                    batch_size=args.batch_size,
                    lr=current_lr,
                    device=device,
                    mu=args.mu
                )
            elif args.train == "scaffold":
                state, n_k, delta_c, c_i_new = scaffold(
                    global_model,
                    clients[cid],
                    c=server_c,
                    c_i=client_cs[cid],
                    epochs=args.local_epochs,
                    batch_size=args.batch_size,
                    lr=current_lr,
                    device=device
                )
                delta_cs.append(delta_c)
                client_cs[cid] = c_i_new

            client_states.append(state)
            client_sizes.append(n_k)

        global_model = server_train(global_model, client_states, client_sizes)

        if args.train == "scaffold":
            server_c = update_server_c(server_c, delta_cs)

        print("\n=== Evaluate global model {0} Round ===".format(r + 1))
        acc, loss = eval(global_model, test_loader, device=device, verbose=False)
        print(f"[{r+1:02d}] acc={acc*100:.2f}%, loss={loss:.6f}")

        prev_lr = current_lr
        scheduler.step(loss)
        current_lr = lr_opt.param_groups[0]["lr"]

        if current_lr != prev_lr:
            print(f"--> LR reduced: {prev_lr:.6g} -> {current_lr:.6g}")

if __name__ == "__main__":
    main()
