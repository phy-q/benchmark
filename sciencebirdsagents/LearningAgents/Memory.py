import os
import random
from collections import namedtuple, deque

import numpy as np

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward', 'is_done'))


class ReplayMemory(object):

    def __init__(self, maxlen):
        self.memory = deque([], maxlen=maxlen)
        self.maxlen = maxlen
        self.action_num = 0

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))
        self.action_num += 1

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

    def to_local(self, path = 'LearningAgents/saved_memory'):
        if not os.path.exists():
            os.mkdir(path)



    def reset(self):
        self.memory = deque(maxlen=self.maxlen)


class PrioritizedReplayMemory(ReplayMemory):
    def __init__(self, maxlen):
        super().__init__(maxlen=maxlen)
        self.priorities = deque(maxlen=maxlen)
        self.reward = deque(maxlen=maxlen)

    def push(self, *args):
        self.memory.append(Transition(*args))
        self.priorities.append(max(self.priorities, default=0.5))
        self.reward.append(args[3])
        self.action_num += 1

    def get_importance(self, probabilities):
        importance = 1 / len(self.memory) * 1 / probabilities
        importance_normalized = importance / max(importance)
        return importance_normalized

    def get_probabilities(self, priority_scale):
        scaled_priorities = np.array(self.priorities) ** priority_scale
        sample_probabilities = scaled_priorities / sum(scaled_priorities)
        return sample_probabilities

    def sample(self, batch_size, priority_scale=1):
        sample_probs = self.get_probabilities(priority_scale=priority_scale)
        sample_indices = random.choices(range(len(self.memory)), k=batch_size,
                                        weights=sample_probs)

        samples = np.array(self.memory, dtype=object)[sample_indices]
        importance = self.get_importance(sample_probs[sample_indices])
        return samples, importance, sample_indices

    def set_priorities(self, indices, errors, offset=0.1):
        for i, e in zip(indices, errors):
            self.priorities[i] = abs(e[0]) + offset

    def reset(self):
        self.memory = deque(maxlen=self.maxlen)
        self.priorities = deque(maxlen=self.maxlen)


class PrioritizedReplayMemoryBalanced(PrioritizedReplayMemory):
    def __init__(self, maxlen):
        super().__init__(maxlen=maxlen)

    def sample(self, batch_size, priority_scale=1):
        sample_probs = self.get_probabilities(priority_scale=priority_scale)
        positive_sample_probs = sample_probs[np.array(self.reward) == 1]
        negative_sample_probs = sample_probs[np.array(self.reward) == 0]
        try:
            positive_sample_indices = random.choices(np.where(np.array(self.reward) == 1)[0], k=batch_size // 2,
                                                     weights=positive_sample_probs)
            negative_sample_indices = random.choices(np.where(np.array(self.reward) == 0)[0], k=batch_size // 2,
                                                     weights=negative_sample_probs)
            sample_indices = np.concatenate((positive_sample_indices, negative_sample_indices), 0)
        except IndexError:
            sample_indices = random.choices(range(len(self.memory)), k=batch_size,
                                            weights=sample_probs)
        samples = np.array(self.memory, dtype=object)[sample_indices]
        importance = self.get_importance(sample_probs[sample_indices])
        return samples, importance, sample_indices


class PrioritizedReplayMemorySumTree(ReplayMemory):
    def __init__(self, maxlen):
        super().__init__(maxlen=maxlen)
        self.PER_e = 0.01  # Hyperparameter that we use to avoid some experiences to have 0 probability of being taken
        self.PER_a = 0.6  # Hyperparameter that we use to make a tradeoff between taking only exp with high priority and sampling randomly
        self.PER_b = 0.4  # importance-sampling, from initial value increasing to 1

        self.PER_b_increment_per_sampling = 0.001

        self.absolute_error_upper = 1.  # clipped abs error
        self.tree = SumTree(maxlen)

    def push(self, *args):
        # Find the max priority
        max_priority = np.max(self.tree.tree[-self.tree.capacity:])

        # If the max priority = 0 we can't put priority = 0 since this experience will never have a chance to be selected
        # So we use a minimum priority
        if max_priority == 0:
            max_priority = self.absolute_error_upper

        self.tree.add(max_priority, Transition(*args))  # set the max priority for new priority
        self.action_num += 1

    def sample(self, batch_size):
        # Create a minibatch array that will contains the minibatch
        minibatch = []

        b_idx = np.empty((batch_size,), dtype=np.int32)

        # Calculate the priority segment
        # Here, as explained in the paper, we divide the Range[0, ptotal] into n ranges
        priority_segment = self.tree.total_priority / batch_size  # priority segment

        for i in range(batch_size):
            # A value is uniformly sample from each range
            a, b = priority_segment * i, priority_segment * (i + 1)
            value = np.random.uniform(a, b)

            # Experience that correspond to each value is retrieved
            index, priority, data = self.tree.get_leaf(value)

            b_idx[i] = index

            minibatch.append(data)
        # set importance to be one
        return np.array(b_idx), np.array(minibatch), np.ones(len(b_idx))

    def set_priorities(self, tree_idx, errors):
        abs_errors = np.abs(errors)
        abs_errors += self.PER_e  # convert to abs and avoid 0
        clipped_errors = np.minimum(abs_errors, self.absolute_error_upper)
        ps = np.power(clipped_errors, self.PER_a)

        for ti, p in zip(tree_idx, ps):
            self.tree.update(ti, p)

    def __len__(self):
        return self.tree.data_pointer


class SumTree(object):
    # from https://pylessons.com/CartPole-PER/
    data_pointer = 0

    # Here we initialize the tree with all nodes = 0, and initialize the data with all values = 0
    def __init__(self, capacity):
        # Number of leaf nodes (final nodes) that contains experiences
        self.capacity = capacity

        # Generate the tree with all nodes values = 0
        # To understand this calculation (2 * capacity - 1) look at the schema below
        # Remember we are in a binary node (each node has max 2 children) so 2x size of leaf (capacity) - 1 (root node)
        # Parent nodes = capacity - 1
        # Leaf nodes = capacity
        self.tree = np.zeros(2 * capacity - 1)

        # Contains the experiences (so the size of data is capacity)
        self.data = np.zeros(capacity, dtype=object)

    def add(self, priority, data):
        # Look at what index we want to put the experience
        tree_index = self.data_pointer + self.capacity - 1

        """ tree:
                    0
                   / \
                  0   0
                 / \ / \
        tree_index  0 0  0  We fill the leaves from left to right
        """

        # Update data frame
        self.data[self.data_pointer] = data

        # Update the leaf
        self.update(tree_index, priority)

        # Add 1 to data_pointer
        self.data_pointer += 1

        if self.data_pointer >= self.capacity:  # If we're above the capacity, we go back to first index (we overwrite)
            self.data_pointer = 0

    def update(self, tree_index, priority):
        # Change = new priority score - former priority score
        change = priority - self.tree[tree_index]
        self.tree[tree_index] = priority

        # then propagate the change through tree
        # this method is faster than the recursive loop
        while tree_index != 0:
            tree_index = (tree_index - 1) // 2
            self.tree[tree_index] += change

    def get_leaf(self, v):
        parent_index = 0

        while True:
            left_child_index = 2 * parent_index + 1
            right_child_index = left_child_index + 1

            # If we reach bottom, end the search
            if left_child_index >= len(self.tree):
                leaf_index = parent_index
                break
            else:  # downward search, always search for a higher priority node
                if v <= self.tree[left_child_index]:
                    parent_index = left_child_index
                else:
                    v -= self.tree[left_child_index]
                    parent_index = right_child_index

        data_index = leaf_index - self.capacity + 1

        return leaf_index, self.tree[leaf_index], self.data[data_index]

    @property
    def total_priority(self):
        return self.tree[0]  # Returns the root node
