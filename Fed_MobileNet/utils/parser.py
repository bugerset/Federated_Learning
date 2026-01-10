import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Federate Learning MobileNet CIFAR-10")

    # seed, device, train function setting
    parser.add_argument("--seed", type=int, default=845)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--train", type=str, default="fedavg", choices=["fedavg", "fedsgd", "fedprox"])
    parser.add_argument("--mu", type=float, default=0.1)

    # dataset setting
    parser.add_argument("--data-root", type=str, default="./data")
    parser.add_argument("--augment", action="store_true")
    parser.add_argument("--normalize", dest="normalize", action="store_true", default=True)
    parser.add_argument("--no-normalize", dest="normalize", action="store_false")
    parser.add_argument("--test-batch-size", type=int, default=128)

    # Client, Batch, Local Epochs, Communicate rounds setting
    parser.add_argument("--num-clients", type=int, default=10)
    parser.add_argument("--client-frac", type=float, default=0.25)
    parser.add_argument("--local-epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--rounds", type=int, default=10)

    # IID, N-IID setting
    parser.add_argument("--partition", type=str, default="iid", choices=["iid", "niid"])
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--min-size", type=int, default=10)
    parser.add_argument("--print-labels", dest="print_labels", action="store_true", default=True)
    parser.add_argument("--no-print-labels", dest="print_labels", action="store_false")

    return parser.parse_args()
