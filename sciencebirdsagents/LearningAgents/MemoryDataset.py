import ast
import logging
import os.path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


class MemoryDatasetMemmap(Dataset):
    def __init__(self, h, w, c, path, **kwargs):
        self.h = h
        self.w = w
        self.c = c
        self.meta = None
        self.path = path
        self.loading_count = 0  # if loading count greater than 10000 reload the dataset
        # if there're pt file and meta file, it means last time the game didn't update
        # we can call update and keep using it
        self.logger = kwargs['logger'] if 'logger' in kwargs else None
        self.learn_last = kwargs['max_len'] if 'max_len' in kwargs else 1e10

        if not os.path.exists(path):
            os.mkdir(path)
        if os.path.exists(os.path.join(self.path, '../state_set.memmap')) and os.path.exists(
                os.path.join(self.path, '../nextstate_set.memmap')) and os.path.exists(
            os.path.join(self.path, 'memory_meta.csv')):
            self.meta = pd.read_csv(os.path.join(self.path, 'memory_meta.csv'))
            self.update()

        else:
            if self.logger:
                self.logger.info('empty dataset')

        self.__remove_all_saved_pt()

    def __remove_all_saved_pt(self):
        # delet all the states and next states that are already saved into the array
        for file in os.listdir(self.path):
            if file[-2:] == 'pt':
                os.remove(os.path.join(self.path, file))

    def __len__(self):
        return len(self.meta) if self.learn_last > len(self.meta) else self.learn_last

    def __getitem__(self, idx):
        if self.loading_count >= 10000:
            del self.state_array
            del self.nextstate_array
            self.state_array = np.memmap(os.path.join(self.path, '../state_set.memmap'), dtype='uint8', mode='r',
                                         shape=(len(self.meta), self.c, self.h, self.w))
            self.nextstate_array = np.memmap(os.path.join(self.path, '../nextstate_set.memmap'), dtype='uint8',
                                             mode='r', shape=(len(self.meta), self.c, self.h, self.w))
            self.loading_count = 0

        if len(self.meta) > self.learn_last:
            # map index to be between len(self.meta) - self.learn_last - 1 to len(self.meta) - 1
            lower = len(self.meta) - self.learn_last - 1
            idx = lower + int(idx/(len(self.meta)-1) * self.learn_last)

        data = self.meta.iloc[idx]
        action, reward, is_done = data.action, data.reward, data.is_done
        action = np.array(ast.literal_eval(action)) if isinstance(action, str) else action
        state = self.state_array[idx]
        next_state = self.nextstate_array[idx]
        self.loading_count += 1

        return state, action, next_state, reward, is_done

    def update(self):
        self.loading_count = 0
        meta = pd.read_csv(os.path.join(self.path, 'memory_meta.csv'))

        try:
            del self.state_array
            del self.nextstate_array
        except AttributeError:
            pass

        if os.path.exists(os.path.join(self.path, '../state_set.memmap')) and os.path.exists(
                os.path.join(self.path, '../nextstate_set.memmap')):

            self.state_array = np.memmap(os.path.join(self.path, '../state_set.memmap'), dtype='uint8', mode='r+',
                                         shape=(len(meta), self.c, self.h, self.w))
            self.nextstate_array = np.memmap(os.path.join(self.path, '../nextstate_set.memmap'), dtype='uint8',
                                             mode='r+', shape=(len(meta), self.c, self.h, self.w))

            for idx in tqdm(range(len(self.meta), len(meta))):
                data = meta.iloc[idx]
                state_path, nextstate_path = data.state_path, data.nextstate_path
                state = torch.load(state_path)
                next_state = torch.load(nextstate_path)
                self.state_array[idx] = torch.Tensor(state)
                self.nextstate_array[idx] = torch.Tensor(next_state)

        else:

            self.state_array = np.memmap(os.path.join(self.path, '../state_set.memmap'), dtype='uint8', mode='w+',
                                         shape=(len(meta), self.c, self.h, self.w))
            self.nextstate_array = np.memmap(os.path.join(self.path, '../nextstate_set.memmap'), dtype='uint8',
                                             mode='w+',
                                             shape=(len(meta), self.c, self.h, self.w))
            for idx in tqdm(range(len(meta))):
                data = meta.iloc[idx]
                state_path, nextstate_path = data.state_path, data.nextstate_path
                state = torch.load(state_path)
                next_state = torch.load(nextstate_path)
                self.state_array[idx] = state
                self.nextstate_array[idx] = next_state

        self.meta = meta
        self.state_array.flush()
        self.nextstate_array.flush()
        self.__remove_all_saved_pt()

        ## return the weight info
        class_count = self.meta['reward'].value_counts()
        if len(class_count) == 2:
            fail_weight = 1/(class_count[0]/sum(class_count))
            pass_weight = 1/(class_count[1]/sum(class_count))
            sample_weight = self.meta['reward'].apply(lambda x: fail_weight if x==0 else pass_weight)

            return sample_weight

    def convert_memmap_worker(self, idx, state_array, nextstate_array, state_path, nextstate_path):
        state = torch.load(state_path)
        next_state = torch.load(nextstate_path)
        state_array[idx] = state
        nextstate_array[idx] = next_state


class MemoryDatasetInRam(Dataset):
    def __init__(self, h, w, c, path):
        self.meta = pd.read_csv(os.path.join(path, 'memory_meta.csv'))
        # save into memory
        self.number_to_save = 12000
        self.path = path
        self.state_tensor = torch.empty((self.number_to_save, c, h, w))
        self.nextstate_tensor = torch.empty((self.number_to_save, c, h, w))
        self.meta = self.meta.iloc[-self.number_to_save:]
        self.meta_new = None
        self.pointer = 0
        for idx in range(len(self.meta)):
            data = self.meta.iloc[idx]
            state_path, nextstate_path = data.state_path, data.nextstate_path
            state = torch.load(state_path)
            next_state = torch.load(nextstate_path)
            self.state_tensor[idx] = torch.Tensor(state)
            self.nextstate_tensor[idx] = torch.Tensor(next_state)

    def update(self):
        self.meta_new = pd.read_csv(os.path.join(self.path, 'memory_meta.csv'))

        if len(self.meta_new) <= self.number_to_save:
            for idx in range(len(self.meta), len(self.meta_new)):
                data = self.meta_new.iloc[idx]
                state_path, nextstate_path = data.state_path, data.nextstate_path
                state = torch.load(state_path)
                next_state = torch.load(nextstate_path)
                self.state_tensor[idx] = torch.Tensor(state)
                self.nextstate_tensor[idx] = torch.Tensor(next_state)

        else:
            for idx in range(len(self.meta), len(self.meta_new)):
                data = self.meta_new.iloc[idx]
                state_path, nextstate_path = data.state_path, data.nextstate_path
                state = torch.load(state_path)
                next_state = torch.load(nextstate_path)
                self.state_tensor[self.pointer] = torch.Tensor(state)
                self.nextstate_tensor[self.pointer] = torch.Tensor(next_state)
                self.pointer += 1
                if self.pointer >= self.number_to_save:
                    self.pointer = 0

        self.meta = self.meta_new

    def __len__(self):
        return len(self.meta[-self.number_to_save:])

    def __getitem__(self, idx):
        data = self.meta.iloc[idx]
        state_path, action, nextstate_path, reward, is_done = data.state_path, data.action, data.nextstate_path, data.reward, data.is_done
        state = self.state_tensor[idx]
        next_state = self.nextstate_tensor[idx]
        return state, action, next_state, reward, is_done


class MemoryDataset(Dataset):
    def __init__(self, path, h=None, w=None, c=None):
        self.meta = pd.read_csv(os.path.join(path, 'memory_meta.csv'))
        self.path = path

    def __len__(self):
        return len(self.meta)

    def __getitem__(self, idx):
        data = self.meta.iloc[idx]
        state_path, action, nextstate_path, reward, is_done = data.state_path, data.action, data.nextstate_path, data.reward, data.is_done
        state = torch.load(state_path)
        next_state = torch.load(nextstate_path)
        return state, action, next_state, reward, is_done

    def update(self):
        self.meta = pd.read_csv(os.path.join(self.path, 'memory_meta.csv'))


if __name__ == '__main__':
    os.chdir("../")
    path = 'LearningAgents/saved_memory/'
    md = MemoryDatasetMemmap(120, 160, c=12, path=path)
    md.update()
    test_loader = DataLoader(md, batch_size=8, pin_memory=True, shuffle=True, num_workers=0)
    for batch in test_loader:
        state, action, next_state, reward, is_done = batch
        print(state.size())
        print(action.size())
        print(next_state.size())
        print(reward.size())
        print(is_done.size())
