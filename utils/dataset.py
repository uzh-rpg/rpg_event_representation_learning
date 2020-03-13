import numpy as np
from os import listdir
from os.path import join

def random_shift_events(events, max_shift=20, resolution=(180, 240)):
    H, W = resolution
    shifted_events = np.copy(events)
    
    x_shift, y_shift = np.random.randint(-max_shift, max_shift+1, size=(2,))
    shifted_events[:,0] += x_shift
    shifted_events[:,1] += y_shift

    valid_events = (shifted_events[:,0] >= 0) & (shifted_events[:,0] < W) & (shifted_events[:,1] >= 0) & (shifted_events[:,1] < H)
    shifted_events = shifted_events[valid_events]

    return shifted_events

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