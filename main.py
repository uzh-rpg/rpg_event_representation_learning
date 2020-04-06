DEBUG = 8

import argparse
import numpy as np
import os
from os.path import dirname
import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter
import tqdm

if DEBUG>0:
    from utils.models1 import Classifier
else:
    from utils.models import Classifier
from utils.dataset import NCaltech101
from utils.loader import Loader
from utils.train_eval import train_one_epoch, eval_one_epoch

if DEBUG==9:
    torch.manual_seed(1)
    np.random.seed(1)

def FLAGS():
    parser = argparse.ArgumentParser("""Train classifier using a learnt quantization layer.""")

    # training / validation dataset
    parser.add_argument("--validation_dataset", default="", required=True)
    parser.add_argument("--training_dataset", default="", required=True)

    # logging options
    parser.add_argument("--log_dir", default="", required=True)

    # loader and device options
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--pin_memory", type=bool, default=True)
    parser.add_argument("--batch_size", type=int, default=4)

    parser.add_argument("--num_epochs", type=int, default=30)
    parser.add_argument("--save_every_n_epochs", type=int, default=5)

    flags = parser.parse_args()

    assert os.path.isdir(dirname(flags.log_dir)), f"Log directory root {dirname(flags.log_dir)} not found."
    assert os.path.isdir(flags.validation_dataset), f"Validation dataset directory {flags.validation_dataset} not found."
    assert os.path.isdir(flags.training_dataset), f"Training dataset directory {flags.training_dataset} not found."

    print(f"----------------------------\n"
          f"Starting training with \n"
          f"num_epochs: {flags.num_epochs}\n"
          f"batch_size: {flags.batch_size}\n"
          f"device: {flags.device}\n"
          f"log_dir: {flags.log_dir}\n"
          f"training_dataset: {flags.training_dataset}\n"
          f"validation_dataset: {flags.validation_dataset}\n"
          f"num_workers: {flags.num_workers}\n"
          f"pin_memory: {flags.pin_memory}\n"
          f"----------------------------")

    return flags

if __name__ == '__main__':
    flags = FLAGS()

    # datasets, add augmentation to training set
    training_dataset = NCaltech101(flags.training_dataset, augmentation=True)
    validation_dataset = NCaltech101(flags.validation_dataset)

    # construct loader, handles data streaming to gpu
    training_loader = Loader(training_dataset, flags, device=flags.device)
    validation_loader = Loader(validation_dataset, flags, device=flags.device)

    # model, and put to device
    model = Classifier(device=flags.device)
    model = model.to(flags.device)

    # optimizer and lr scheduler
    optimizerSelect = 'adam'
    if optimizerSelect == 'sgd':
        optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4, momentum=0.9, weight_decay=1e-5)
    elif optimizerSelect == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.5)

    iteration = 0
    min_validation_loss = 1000

    for i in range(flags.num_epochs):
        
        print(f"Training step [{i:3d}/{flags.num_epochs:3d}]")
        training_loss, training_accuracy, iteration = train_one_epoch(model, flags.device, optimizer, training_loader, iteration)
        print(f"Training Iteration {iteration:5d}  Loss {training_loss:.4f}  Accuracy {training_accuracy:.4f}")

        if i%10 == 9:
            if optimizerSelect == 'adam':
                lr_scheduler.step()

        if i%5 == 0:
            print(f"Validation step [{i:3d}/{flags.num_epochs:3d}]")
            validation_loss, validation_accuracy = eval_one_epoch(model, flags.device, validation_loader)
            print(f"Validation Loss {validation_loss:.4f}  Accuracy {validation_accuracy:.4f}")
        
            if validation_loss < min_validation_loss:
                min_validation_loss = validation_loss
                state_dict = model.state_dict()
                torch.save({
                    "state_dict": state_dict,
                    "min_val_loss": min_validation_loss,
                    "iteration": iteration
                }, "log/model_best.pth")
                print("New best at ", validation_loss)
            if i % flags.save_every_n_epochs == 0:
                state_dict = model.state_dict()
                torch.save({
                    "state_dict": state_dict,
                    "min_val_loss": min_validation_loss,
                    "iteration": iteration
                }, "log/checkpoint_%05d_%.4f.pth" % (iteration, min_validation_loss))