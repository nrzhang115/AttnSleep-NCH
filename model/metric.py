import torch
from sklearn.metrics import confusion_matrix, f1_score


def accuracy(output, target):
    with torch.no_grad():
        pred = torch.argmax(output, dim=1)
        assert pred.shape[0] == len(target)
        correct = 0
        correct += torch.sum(pred == target).item()
    return correct / len(target)


def f1(output, target):
    with torch.no_grad():
        pred = torch.argmax(output, dim=1)
        assert pred.shape[0] == len(target)
        # Calculate confusion matrix
        conf_matrix = confusion_matrix(target.cpu().numpy(), pred.cpu().numpy())
        # Calculate support for each class
        supports = conf_matrix.sum(axis=1)
        print(f"Supports for each class: {supports}")
    return f1_score(pred.cpu().numpy(), target.data.cpu().numpy(), average='macro')
