import math
import numpy as np
from os import listdir
from os.path import join
import random

def random_shift_events(events, max_shift=20, resolution=(180, 240)):
    H, W = resolution
    x_shift, y_shift = np.random.randint(-max_shift, max_shift+1, size=(2,))
    events[:,0] += x_shift
    events[:,1] += y_shift

    # Want to rotate events, not done
    # max_rad = 0.18
    # x, y, t, p = events.T
    # x_old = x.copy() - W//2
    # y_old = y.copy() - H//2
    # rotate_rad = ( random.random()-0.5)*max_rad
    # print(rotate_rad)
    # x = x_old*math.cos(rotate_rad) + y_old*math.sin(rotate_rad) + W//2
    # y = -1*x_old*math.sin(rotate_rad) + y_old*math.cos(rotate_rad) + H//2
    # events = np.stack( (x, y, t, p), axis=1)

    valid_events = (events[:,0] >= 0) & (events[:,0] < W) & (events[:,1] >= 0) & (events[:,1] < H)
    events = events[valid_events]

    return events

def random_flip_events_along_x(events, resolution=(180, 240), p=0.5):
    H, W = resolution
    if np.random.random() < p:
        events[:,0] = W - 1 - events[:,0]
    return events

class NCaltech101:
    def __init__(self, root, augmentation=False):
        self.classes = listdir(root)

        self.files = []
        self.labels = []

        self.augmentation = augmentation

        for i, c in enumerate(self.classes):
            new_files = [join(root, c, f) for f in listdir(join(root, c))]
            self.files += new_files
            self.labels += [i] * len(new_files)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        """
        returns events and label, loading events from aedat
        :param idx:
        :return: x,y,t,p,  label
        """
        label = self.labels[idx]
        f = self.files[idx]
        events = np.load(f).astype(np.float32)

        if self.augmentation:
            events = random_shift_events(events)
            events = random_flip_events_along_x(events)

        return events, label