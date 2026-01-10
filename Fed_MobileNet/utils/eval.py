import torch

def eval(model, dataloader, device="cpu", verbose=True):
    model.eval()

    loss_fn = torch.nn.CrossEntropyLoss(reduction="sum")
    size = len(dataloader.dataset)
    loss, correct = 0, 0

    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            pred = model(x)
            loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    loss /= size
    correct /= size

    if verbose:
        print(f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {loss:>8f}")

    return correct, loss