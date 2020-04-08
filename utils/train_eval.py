import numpy as np
import torch
import tqdm
from .loss import cross_entropy_loss_and_accuracy

def train_one_epoch(model, device, optimizer, dataloader, iteration):
    sum_accuracy = 0
    sum_loss = 0
    
    for events, labels in tqdm.tqdm(dataloader):
        labels = labels.to(device)
        
        optimizer.zero_grad()

        pred_labels, representation = model(events)
        loss, accuracy = cross_entropy_loss_and_accuracy(pred_labels, labels)

        loss.backward()
        optimizer.step()

        sum_accuracy += accuracy
        sum_loss += loss

        iteration += 1

    training_loss = sum_loss.item() / len(dataloader)
    training_accuracy = sum_accuracy.item() / len(dataloader)
    return training_loss, training_accuracy, iteration

def eval_one_epoch(model, device, dataloader):
    sum_accuracy = 0
    sum_loss = 0
    
    for events, labels in tqdm.tqdm(dataloader):
        labels = labels.to(device)
        
        with torch.no_grad():
            pred_labels, representation = model(events)
            loss, accuracy = cross_entropy_loss_and_accuracy(pred_labels, labels)

        sum_accuracy += accuracy
        sum_loss += loss

    validation_loss = sum_loss.item() / len(dataloader)
    validation_accuracy = sum_accuracy.item() / len(dataloader)
    return validation_loss, validation_accuracy
