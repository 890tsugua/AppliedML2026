import random
from collections import defaultdict
from torch.utils.data import Sampler


class AtLeastOnePerClassBatchSampler(Sampler):
    def __init__(self, labels, batch_size, drop_last=True):
        self.labels = list(labels)
        self.batch_size = batch_size
        self.drop_last = drop_last

        self.class_to_indices = defaultdict(list)
        for idx, label in enumerate(self.labels):
            self.class_to_indices[label].append(idx)

        self.classes = list(self.class_to_indices.keys())
        self.num_classes = len(self.classes)

        if batch_size < self.num_classes:
            raise ValueError(
                f"batch_size={batch_size} must be >= number of classes={self.num_classes}"
            )

        self.num_batches = len(self.labels) // batch_size

    def __iter__(self):
        all_indices = list(range(len(self.labels)))

        for _ in range(self.num_batches):
            batch = []

            # guarantee one from each class
            used = set()
            for cls in self.classes:
                idx = random.choice(self.class_to_indices[cls])
                batch.append(idx)
                used.add(idx)

            # fill remaining slots randomly
            remaining_slots = self.batch_size - self.num_classes
            candidates = [idx for idx in all_indices if idx not in used]

            batch.extend(random.sample(candidates, remaining_slots))

            random.shuffle(batch)
            yield batch

    def __len__(self):
        return self.num_batches