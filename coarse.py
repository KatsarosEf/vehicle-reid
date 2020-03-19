from __future__ import print_function, division
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import datasets, models, transforms
import torch.nn.functional as F
import os
from sampler import BalancedBatchSampler as BatchSampler
from evaluation import compute_rank
from utils import BatchTripletSelector
import copy
from torch.utils.tensorboard import SummaryWriter


print("PyTorch Version: ",torch.__version__)
print("Torchvision Version: ",torchvision.__version__)

def dataset(data_dir, input_size = 224):
    "Initializes dataset dictionaries."    
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize(input_size),
            transforms.RandomRotation(3),
            transforms.ToTensor(),
            transforms.Normalize([-0.4246, -0.2174,  0.0104], [1.0356, 1.0465, 1.0460])
        ]),
        'val': transforms.Compose([
            transforms.Resize(input_size),
            transforms.ToTensor(),
            transforms.Normalize([-0.4246, -0.2174,  0.0104], [1.0356, 1.0465, 1.0460])
        ]),
    }    
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['train', 'val']}
    return(image_datasets)

def dataloader(image_datasets, batch_size, num_workers):  
    "Initializes dataloaders."  
    dataloaders = {}
    train_sampler = BatchSampler(image_datasets['train'], 6, 5)
    dataloaders['train']  = torch.utils.data.DataLoader(image_datasets['train'], batch_sampler = train_sampler, num_workers = num_workers)
    dataloaders['val'] =  torch.utils.data.DataLoader(image_datasets['val'], batch_size = batch_size, num_workers = num_workers)
    return(dataloaders)


def train_model(model, dataloaders, criterion, optimizer, nr_epochs, scheduler, string_name, device):
    "Model training for 'nr_epochs' epochs."
    logger = SummaryWriter()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_rank1_v, best_rank5_v = 0.0, 0.0
    feats_val, labels_val = torch.ones(len(dataloaders['val'].dataset), 1024), torch.ones(len(dataloaders['val'].dataset))

    for epoch in range(nr_epochs):
        print('Epoch {}/{}'.format(epoch+1, nr_epochs))
        print('-' * 10)       
        for phase in ['train', 'val']:
            # configure model functionality - train/val
            if phase == 'train':
                model.train()
                running_loss, nr_batch_triplets = (0.0, 0)
            else:
                model.eval()  
            
            for index, (inputs, labels) in enumerate(dataloaders[phase]):

                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                embeddings = F.adaptive_avg_pool2d(F.relu(model.features(inputs), inplace=True), (1, 1)).view(inputs.size(0), -1)
                #for each batch update
                if phase == 'train':
                    with torch.set_grad_enabled(phase=='train'):        
                        batch_triplets = triplet_selector.get_triplets(embeddings, labels)
                        nr_batch_triplets += batch_triplets.size(0)
                        loss = criterion(embeddings[batch_triplets[:,0]], embeddings[batch_triplets[:,1]], embeddings[batch_triplets[:,2]])                                                
                        running_loss += batch_triplets.size(0)*loss.item() 
                        loss.backward()
                        optimizer.step()
                else:
                    with torch.no_grad():
                        feats_val[index*dataloaders[phase].batch_size:dataloaders[phase].batch_size*(index+1)] = embeddings
                        labels_val[index*dataloaders[phase].batch_size:dataloaders[phase].batch_size*(index+1)] = labels
            # for each epoch
            if phase == 'train':
                epoch_TL = running_loss/ nr_batch_triplets                
                print('{} Triplet Loss: {:.4f} Informative triplets: {:.4f}'.format(phase, epoch_TL, round(nr_batch_triplets/len(dataloaders[phase]))))
            else:
                rank1_v, rank5_v = compute_rank(feats_val, labels_val, dataloaders[phase], device)
                print('{} Rank-1: {:.4f} Rank-5: {:.4f}'.format(phase, rank1_v, rank5_v))
                scheduler.step()
            # deep copy the model
            if phase == 'val' and rank1_v > best_rank1_v:
                best_rank1_v, best_rank5_v = rank1_v, rank5_v            
                best_model_wts = copy.deepcopy(model.state_dict())
                torch.save(model.state_dict(), string_name + '.pt')
        
        logger.add_scalars(string_name, {'train_loss':running_loss/nr_batch_triplets,
                                         'nr_triplets':nr_batch_triplets,
                                         'rank1_val': rank1_v,
                                         'rank5_val': rank5_v}, epoch+1)
    logger.close()     
    print('Training Finished. Best Validation Rank-1: {:4f} and Best val Rank-5: {:4f} '.format(best_rank1_v, best_rank5_v))
    model.load_state_dict(best_model_wts)
    return model


def main(string_name, nr_epochs):

    #data_dir = '/home/efklidis/data/reranking_data/'
    data_dir = '/home/efklidis/data/224x224/toy/'

    print("Initializing dataset...")
    image_datasets  = dataset(data_dir)    
    print("Dataset is loaded.")
    
    print("\nInitializing dataloaders...")
    dataloaders = dataloader(image_datasets, batch_size = 8, num_workers = 4)
    print("Dataloaders are booted.")

    print("\nInitializing model...")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = models.densenet121(pretrained=True).to(device)    
    print("Model is mounted on device:", str(device))

    print("\nInitializing optimizer...")  
    criterion = nn.TripletMarginLoss(margin=0.3, reduction='mean')
    optimizer = optim.Adam(model.parameters(), lr=0.00003, weight_decay=0.001) #initially at lr=0.0003
    scheduler =  optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50, 80, 110, 140, 160, 180, 200, 220, 240, 260, 280], gamma=0.9)
    print("Set.")

    print("\nTraining for ", str(nr_epochs), "epochs...")
    model = train_model(model, dataloaders, criterion, optimizer, nr_epochs, scheduler, string_name, device)
    

if __name__ == '__main__':
    triplet_selector = BatchTripletSelector(0.3)
    main(string_name = 'triplet_batchall', nr_epochs = 30)
