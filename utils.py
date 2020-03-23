from itertools import combinations
import numpy as np
import torch


def get_lr(optimizer):
    """
    Returns the learning rate related to the optimizer status.
    """
    for param_group in optimizer.param_groups:
        return param_group['lr']

def create_logs(string_name):
    """
    Initialize logger text file.
    """
    with open(string_name + '.txt', 'w') as file:
        file.write("")
        file.close()
        
def update_logs(string_name, phase, epoch, num_epochs, optimizer, train_history, val_history):
    """
    Update logger text file.
    """
    f = open(string_name + ".txt", "a")
    if phase == 'train':
        f.write('Epoch {}/{}'.format(epoch+1, num_epochs ) + ' -' * 20 + ' \n')
        f.write('Learning rate for this epoch was ' + str(get_lr(optimizer))  + ' \n')
        f.write('Train Triplet loss: ' + str(train_history[epoch][0]) + 'Informative triplets: ' + str(train_history[epoch][1]))
    else:
        f.write('Val Rank-1 Accuracy: '+ str(val_history[epoch][0]) + ', Rank-5 Accuracy:' + str(val_history[epoch][1]) + ' \n' )
        f.close()


def pdist(vectors):
    """
    Computes euclidian pairwise distances of n data points(embeddings).
    Args:
        vectors (torch.tensor): A (nxd) embeddings tensor.
    Returns:
        An L2 pairwise (nxn) distance matrix.
    """
    distance_matrix = -2 * vectors.mm(torch.t(vectors)) + vectors.pow(2).sum(dim=1).view(1, -1) + vectors.pow(2).sum(
        dim=1).view(-1, 1)
    return distance_matrix

class BatchAll:
    """
    Initializes the core sampler object. Yields all possible triplet combinations given a batch sample (embeddings and labels) and a triplet constraint.
    """
    def __init__(self, margin):
        super(BatchAll, self).__init__()
        self.margin = margin


    def get_triplets(self, embeddings, labels):
        distance_matrix = pdist(embeddings)
        labels = labels.cpu().data.numpy()
        triplets = []

        for label in set(labels):
            label_mask = (labels == label)
            label_indices = np.where(label_mask)[0]
            if len(label_indices) < 2:
                continue
            negative_indices = np.where(np.logical_not(label_mask))[0]
            anchor_positives = np.array(list(combinations(label_indices, 2)))

            ap_distances = distance_matrix[anchor_positives[:, 0], anchor_positives[:, 1]]
            for anchor_positive, ap_distance in zip(anchor_positives, ap_distances):
                loss_values = ap_distance - distance_matrix[torch.LongTensor(np.array([anchor_positive[0]])), torch.LongTensor(negative_indices)] + self.margin
                loss_values = loss_values.data.cpu().numpy()
                hard_negative = np.where(loss_values > 0)[0]
                
                if hard_negative is not None:
                    hard_negative = negative_indices[hard_negative]
                    trips_to_add = np.hstack(( np.tile(anchor_positive[0:2], (len(hard_negative), 1)), hard_negative.reshape(len(hard_negative), 1) )).tolist()
                    triplets.extend(trips_to_add)
        if len(triplets) == 0:
            triplets.append([anchor_positive[0], anchor_positive[1], negative_indices[0]])


        return torch.LongTensor(triplets)

    def pdist(self, vectors):
        """
        Computes euclidian pairwise distances of n data points(embeddings).
        """
        distance_matrix = -2 * vectors.mm(torch.t(vectors)) + vectors.pow(2).sum(dim=1).view(1, -1) + vectors.pow(2).sum(
            dim=1).view(-1, 1)
        return distance_matrix