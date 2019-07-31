import torch


def cross_entropy_loss_and_accuracy(prediction, target):
    cross_entropy_loss = torch.nn.CrossEntropyLoss()
    loss = cross_entropy_loss(prediction, target)
    accuracy = (prediction.argmax(1) == target).float().mean()
    return loss, accuracy