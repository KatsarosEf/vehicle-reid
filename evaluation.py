import torch
import numpy as np

def pdist_torch(emb1, emb2):
    '''
    compute the eucilidean distance matrix between embeddings1 and embeddings2
    using gpu
    '''
    m, n = emb1.shape[0], emb2.shape[0]
    emb1_pow = torch.pow(emb1, 2).sum(dim = 1, keepdim = True).expand(m, n)
    emb2_pow = torch.pow(emb2, 2).sum(dim = 1, keepdim = True).expand(n, m).t()
    dist_mtx = emb1_pow + emb2_pow
    dist_mtx = dist_mtx.addmm_(1, -2, emb1, emb2.t())
    dist_mtx = dist_mtx.clamp(min = 1e-12).sqrt()
    return dist_mtx


def pick_random(loader):
    indices = torch.arange(0, len(loader.dataset.imgs))
    labels = torch.LongTensor(loader.dataset.targets)
    images_sel = torch.zeros(len(torch.unique(labels)), dtype=torch.int)
    for label in torch.unique(labels):
        index = torch.randint(len(indices[label ==labels]), (1,))
        images_sel[label] = indices[label ==labels][index]               
    return(images_sel)

def rank_k(distances, k):
    predictions = torch.ones(k, distances.shape[1], dtype=torch.int)
    for column in range(distances.shape[1]):
        predictions[:,column] = distances[:,column].topk(k, sorted = True, largest=False)[1]
    return(predictions)


def belongs(min_k_distances, true_labels):
    correct_k = 0
    for probe in range(len(true_labels)):
        if true_labels[probe].item() in min_k_distances[:,probe]:
            correct_k += 1
    return(correct_k)


def compute_rank(features, labels, test_loader, device, replications=5):

    rank_1_replicates, rank_5_replicates = ([], [])
    
    with torch.no_grad():
        for replication in range(replications):    

            torch.manual_seed(1992)
            gallery = pick_random(test_loader)
            probe = torch.from_numpy( np.setdiff1d(torch.arange(len(test_loader.dataset)), gallery)).long().to(device)
            gallery_feats = features[gallery.long()].to(device)
            probe_feats, probe_targets = features[probe].to(device), labels[probe].to(device)
            distances = pdist_torch(gallery_feats, probe_feats)
            
            correct1, correct5 = (0,0)
            correct1 +=  probe_targets.eq(torch.argmin(distances, dim=0).float()).sum().item()
            correct5 += belongs(rank_k(distances, 5), probe_targets)
    
            rank_1, rank_5 = (correct1 / probe_targets.size()[0], correct5 / probe_targets.size()[0])
            rank_1_replicates.append(rank_1)
            rank_5_replicates.append(rank_5)
        
    return(sum(rank_1_replicates)/replications, sum(rank_5_replicates)/replications )

