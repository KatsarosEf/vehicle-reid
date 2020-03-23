import torch
import numpy as np

def pdist_torch(emb1, emb2):
    '''
    Takes two embedding tensors (nxd) and (mxd),
    Returns the (nxm) pairwise euclidean distance matrix using gpu - GitHub repo: CoinCheung/triplet-reid-pytorch.
    '''
    m, n = emb1.shape[0], emb2.shape[0]
    emb1_pow = torch.pow(emb1, 2).sum(dim = 1, keepdim = True).expand(m, n)
    emb2_pow = torch.pow(emb2, 2).sum(dim = 1, keepdim = True).expand(n, m).t()
    dist_mtx = emb1_pow + emb2_pow
    dist_mtx = dist_mtx.addmm_(1, -2, emb1, emb2.t())
    dist_mtx = dist_mtx.clamp(min = 1e-12).sqrt()
    return dist_mtx


def pick_random(loader):
    """
    Splits the validation/test sets according to the evaluation protocol provided with VehicleID dataset.
    Returns the gallery set indices.
    """
    indices = torch.arange(0, len(loader.dataset.imgs))
    labels = torch.LongTensor(loader.dataset.targets)
    images_sel = torch.zeros(len(torch.unique(labels)), dtype=torch.int)
    for label in torch.unique(labels):
        index = torch.randint(len(indices[label ==labels]), (1,))
        images_sel[label] = indices[label ==labels][index]               
    return(images_sel)

def rank_k(distances, k):
    """
    Takes the query-gallery distance matrix and the k referring to rank-k accuracy.
    Returns the top-k predictions for each query.
    """
    predictions = torch.ones(k, distances.shape[1], dtype=torch.int)
    for query in range(distances.shape[1]):
        predictions[:,query] = distances[:,query].topk(k, sorted = True, largest=False)[1]
    return(predictions)


def belongs(min_k_distances, true_labels):
    """
    Takes the top-k predictions and the true labels.
    Returns boolean value determining whether each query's true label belongs or not in the top-k predictions.
    """
    correct_k = 0
    for probe in range(len(true_labels)):
        if true_labels[probe].item() in min_k_distances[:,probe]:
            correct_k += 1
    return(correct_k)


def compute_rank(features, labels, test_loader, device, replications=5):
    """
    Takes features and labels tensors, a dataloader object, the device to perform operations on, and the number of replications.
    Returns rank-1 and rank-5 accuracy averaged over replications.
    """

    rank_1_replicates, rank_5_replicates = ([], [])
    
    with torch.no_grad():
        for replication in range(replications):    

            torch.manual_seed(1992)
            gallery = pick_random(test_loader)
            probe = torch.from_numpy(np.setdiff1d(torch.arange(len(test_loader.dataset)), gallery)).long().to(device)
            gallery_feats = features[gallery.long()].to(device)
            query_feats, query_targets = features[probe].to(device), labels[probe].to(device)
            distances = pdist_torch(gallery_feats, query_feats)
            
            correct1, correct5 = (0,0)
            correct1 +=  query_targets.eq(torch.argmin(distances, dim=0).float()).sum().item()
            correct5 += belongs(rank_k(distances, 5), query_targets)
    
            rank_1, rank_5 = (correct1 / query_targets.size(0), correct5 / query_targets.size(0))
            rank_1_replicates.append(rank_1)
            rank_5_replicates.append(rank_5)
        
    return(sum(rank_1_replicates)/replications, sum(rank_5_replicates)/replications)

