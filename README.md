# Federated Learning (FedAvg / FedSGD / FedProx / SCAFFOLD) on CIFAR-10 and MNIST with MobileNet (PyTorch)

A simple federated learning **simulation** in PyTorch using **MobileNet** on **CIFAR-10** and **MNIST**.  
Supports:
- **FedAvg**   (client local training + weighted model averaging)
- **FedSGD**   (1 epoch with full client batch to simulate gradient-style update)
- **FedProx**  (FedAvg with a proximal regularization term)
- **SCAFFOLD** (SCAFFOLD with control variation)
- **IID** and **Non-IID** partitioning (Dirichlet-based)

## Project Overview

This project simulates the typical federated learning loop:

1. Load CIFAR-10 or MNIST train/test
2. Split training set into multiple client datasets (IID or Non-IID)
3. For each round:
   - Randomly sample a fraction of clients
   - Train each selected client locally
   - Aggregate client weights on the server (weighted by client dataset size)
4. Evaluate the global model on the CIFAR-10 test set each round

## Recommended Folder Structure

Your `main.py` imports modules like `data.cifar10`, `fl.fedavg`, etc.  
So the easiest way to run without changing code is to organize files like this:
```
├── main.py
├── data/
│   ├── __init__.py
│   ├── cifar10.py
│	├── mnist.py
│   └── partition.py
├── fl/
│   ├── __init__.py
│   ├── fedavg.py
│	├── fedprox.py
│	├── scaffold.py
│   └── server.py
├── models/
│   ├── __init__.py
│   └── mobilenet.py
└── utils/
 	├── __init__.py
 	├── device.py
    ├── eval.py
    ├── parser.py
    └── seed.py
```

> If you keep everything in one folder, you must edit imports in `main.py` accordingly.

## Requirements

- Python 3.9+ recommended
- PyTorch + torchvision
- numpy

Run with default settings:
```bash
python main.py
```
Example: FedAvg + IID
```bash
python main.py --train fedavg --partition iid
```
Example: FedAvg + Non-IID (Dirichlet)
```bash
python main.py --train fedavg --partition niid --alpha 0.5 --min-size 10
```
Example: FedProx (with mu)
```bash
python main.py --train fedprox --mu 0.1 --partition niid --alpha 0.5
```

## Device Selection

The code supports:
```
	•	--device auto (default): selects CUDA if available, else MPS (Apple Silicon), else CPU
	•	--device cuda
	•	--device mps
	•	--device cpu
```

Example:
```bash
python main.py --device auto
```

## CLI Arguments

Key arguments (from utils/parser.py):
```
	•	Reproducibility / compute
		•	--seed (default: 845)
		•	--device in {auto,cpu,cuda,mps}

	•	Training method
		•	--train in {fedavg,fedsgd,fedprox,scaffold}
		•	--mu (FedProx proximal strength, default 0.1)

	•	Dataset
		•   --data-set (default cifar10, choices=[cifar10, mnist])
		•	--data-root (default ./data)
		•	--augment / --no-augment
		•	--normalize / --no-normalize
		•	--test-batch-size (default 128)

	•	Federated learning config
		•	--num-clients (default 10)
		•	--client-frac fraction of clients sampled per round (default 0.25)
		•	--local-epochs (default 1)
		•	--batch-size (default 100)
		•	--lr learning rate (default 1e-2)
		•	--rounds communication rounds (default 10)

	•	Data partitioning
		•	--partition in {iid,niid}
		•	--alpha: Dirichlet concentration parameter controlling Non-IID severity.
		    	├── α = 0.1 ~ 0.3: highly skewed label distribution (strong Non-IID)
		  		├──	α = 0.5: moderate Non-IID (default)
		  		└──	α = 0.8 ~ 1.0: closer to IID
		•	--min-size minimum samples per client in non-IID (default 10)
		•	--print-labels / --no-print-labels

	•	Learning rate Scheduler (ReduceOnPlateau)
		•	--lr-factor learning rate * factor (default 0.5)
		•	--lr-patience (default 5)
		•	--min-lr (deafult 1e-6)
		•	--lr-threshold (default 1e-4)
		•	--lr-cooldown (default 0)
```

Notes on Implementation
```
	•	Client training (fl/fedavg.py, fl/fedprox.py, fl/scaffold.py)
	•	Uses SGD with momentum=0.9 and weight decay=5e-4 (In scaffold, no momentum and weight decay)
	•	Returns the local state_dict moved to CPU (for aggregation)
	•	Server aggregation (fl/server.py)
	•	Weighted average of parameters using client dataset sizes
	•	Non-IID partitioning (data/partition.py)
	•	Uses a Dirichlet distribution per class across clients
	•	Includes a safety loop to ensure each client has at least min_size samples
```

## Expected Output

Each round prints evaluation results like:
```bash
=== Evaluate global model 1 Round ===
[01] acc=XX.XX%, loss=Y.YYYYYY
```

With FedAVG, data_set="cifar10", num_clients=100, client_frac=0.25, local_epochs=5, batch_size=50, lr=le-2, rounds=200, partition="niid", alpha=0.4:
<br>91 Round ACC=66.30%, loss=0.972281
<br>113 Round ACC=66.80%, loss=0.975469

<br>With FedProx, data_set="cifar10", num_clients=100, client_frac=0.25, local_epochs=5, batch_sie=50, lr=1e-2, rounds=200, partition="niid", alpha=0.4:
<br>91 Round ACC=59.35%, loss = 1.119520
<br>104 Round ACC=61.32%, loss=1.094311
<br>132 Round ACC=61.96%, loss=1.069016
<br>138 Round ACC=62.38%, loss=1.070478
<br>149 Round ACC=63.00%, loss=1.062149
<br>174 Round ACC=63.16%, loss=1.059052

<br>With SCAFFOLD, data_set="cifar10", num_clients=100, client_frac=0.25, local_epochs=5, batch_sie=50, lr=1e-2, rounds=200, partition="niid", alpha=0.4:
<br>136 Round ACC=62.00%, loss=1.085654
<br>144 Round ACC=63.03%, loss=1.059989
<br>153 Round ACC=64.48%, loss=1.1003411
<br>164 Round ACC=65.30%, loss=0.991139
<br>196 Round ACC=65.80%, loss=0.976644






