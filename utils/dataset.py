import math
import numpy as np
from os import listdir
from os.path import join
import random

def random_shift_rotate_scale_events(events, max_shift=20, max_rad=0.1745, resolution=(180, 240)):
    H, W = resolution

    x, y, t, p = events.T
    rotate_rad = 0. if max_rad==None else (random.random()-0.5)*2*max_rad
    scale = 1.0 if random.random()<0.5 else random.random()/2. + 0.7
    x -= W//2
    y -= H//2
    x_new = ( x*np.cos(rotate_rad) + y*np.sin(rotate_rad)) * scale + W//2
    y_new = ( -1*x*np.sin(rotate_rad) + y*np.cos(rotate_rad)) * scale + H//2
    x_new = np.rint(x_new)
    y_new = np.rint(y_new)
    events = np.stack( (x_new, y_new, t, p), axis=1)

    x_shift, y_shift = np.random.randint(-max_shift, max_shift+1, size=(2,))
    events[:,0] += x_shift
    events[:,1] += y_shift

    valid_events = (events[:,0] >= 0) & (events[:,0] < W) & (events[:,1] >= 0) & (events[:,1] < H)
    events = events[valid_events]

    return events

def random_flip_events_along_x(events, resolution=(180, 240), p=0.5):
    H, W = resolution
    if np.random.random() < p:
        events[:,0] = W - 1 - events[:,0]
    return events

class NCaltech101:
    def __init__(self, root, augmentation=False, resolution=(180, 240)):
        self.classes = listdir(root)
        self.classes.sort()

        self.files = []
        self.labels = []

        self.augmentation = augmentation
        self.resolution = resolution

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
            events = random_shift_rotate_scale_events(events, max_rad=None, resolution=self.resolution)
            events = random_flip_events_along_x(events, resolution=self.resolution)

        return events, label

    def getClasses(self):
        return self.classes.copy()