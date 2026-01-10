# Federated Learning (FedAvg / FedSGD / FedProx) on CIFAR-10 with MobileNet (PyTorch)

A simple federated learning **simulation** in PyTorch using **MobileNet** on **CIFAR-10**.  
Supports:
- **FedAvg**   (client local training + weighted model averaging)
- **FedSGD**   (1 epoch with full client batch to simulate gradient-style update)
- **FedProx**  (FedAvg with a proximal regularization term)
- **IID** and **Non-IID** partitioning (Dirichlet-based)

## Project Overview

This project simulates the typical federated learning loop:

1. Load CIFAR-10 train/test
2. Split training set into multiple client datasets (IID or Non-IID)
3. For each round:
   - Randomly sample a fraction of clients
   - Train each selected client locally
   - Aggregate client weights on the server (weighted by client dataset size)
4. Evaluate the global model on the CIFAR-10 test set each round

## Recommended Folder Structure

Your `main.py` imports modules like `data.cifar10`, `fl.client`, etc.  
So the easiest way to run without changing code is to organize files like this:
```
├── main.py
├── data/
│   ├── init.py
│   ├── cifar10.py
│   └── partition.py
├── fl/
│   ├── init.py
│   ├── client.py
│   └── server.py
├── models/
│   ├── init.py
│   └── mobilenet.py
└── utils/
├── init.py
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
	•	**Reproducibility / compute**
	•	--seed (default: 845)
	•	--device in {auto,cpu,cuda,mps}
	•	**Training method**
	•	--train in {fedavg,fedsgd,fedprox}
	•	--mu (FedProx proximal strength, default 0.1)
	•	**Dataset**
	•	--data-root (default ./data)
	•	--augment (train-time augmentation)
	•	--normalize / --no-normalize
	•	--test-batch-size (default 128)
	•	**Federated learning config**
	•	--num-clients (default 10)
	•	--client-frac fraction of clients sampled per round (default 0.25)
	•	--local-epochs (default 1)
	•	--batch-size (default 100)
	•	--lr learning rate (default 1e-3)
	•	--rounds communication rounds (default 5)
	•	**Data partitioning**
	•	--partition in {iid,niid}
	•	--alpha: Dirichlet concentration parameter controlling Non-IID severity.
		    ├── α = 0.1 ~ 0.3: highly skewed label distribution (strong Non-IID)
		  	├──	α = 0.5: moderate Non-IID
		  	└──	α = 0.8 ~ 1.0: closer to IID
	•	--min-size minimum samples per client in non-IID (default 10)
	•	--print-labels / --no-print-labels
```

Notes on Implementation
```
	•	Client training (fl/client.py)
	•	Uses SGD with momentum=0.9 and weight decay=5e-4
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
=====================================
```
