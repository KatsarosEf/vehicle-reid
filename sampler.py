import numpy as np
from torch.utils.data.sampler import Sampler
from torch.utils.data import DataLoader

class BalancedBatchSampler(Sampler):
    '''
    Initializes the sampler to wrap the dataloader object. Method __iter__ yields the indices each time it is called.
    - GitHub repo: CoinCheung/triplet-reid-pytorch
    '''
    def __init__(self, dataset, nr_class, nr_num):
        super(BalancedBatchSampler, self).__init__(dataset)
        self.nr_class = nr_class
        self.nr_num = nr_num
        self.batch_size = nr_class * nr_num
        loader = DataLoader(dataset)
        labels = []
        for _, label in loader:
            labels.append(label)
        self.labels = np.array(labels)
        self.labels_uniq = np.array(list(set(self.labels)))
        self.len = len(dataset) // self.batch_size
        
        self.label_img_dict = {label: np.where(self.labels == label)[0]
                                 for label in self.labels_uniq}      
        self.nr_iter = len(self.labels_uniq) // self.nr_class

    def __iter__(self):
        running_p = 0
        np.random.shuffle(self.labels_uniq)
        for k, v in self.label_img_dict.items():
            np.random.shuffle(self.label_img_dict[k])
        for i in range(self.nr_iter):
            label_batch = self.labels_uniq[running_p: running_p + self.nr_class]
            running_p += self.nr_class
            idx = []
            for lb in label_batch:
                if len(self.label_img_dict[lb]) > self.nr_num:
                    id_sample = np.random.choice(self.label_img_dict[lb],
                            self.nr_num, replace = False)
                else:
                    id_sample = np.random.choice(self.label_img_dict[lb],
                            self.nr_num, replace = True)
                idx.extend(id_sample.tolist())
            yield idx

    def __len__(self):
        return self.nr_iter
