from __future__ import print_function
from __future__ import division
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import datasets, models, transforms
import torch.nn.functional as F
import os
import torch
import copy
from sampler import BalancedBatchSampler as BatchSampler
from evaluation import compute_rank
from utils import BatchAll
from torch.utils.tensorboard import SummaryWriter
print("PyTorch Version: ",torch.__version__)
print("Torchvision Version: ",torchvision.__version__)


def dataset(data_dir, input_size = (62, 152)):
    """ 
    Loads the transformations required for training the fine model and initializes training and validation datasets.
    Normalization applied here refers to the mean and std of the windshield images.
    Args:
        data_dir (string): Data directory. Each folder within the directory stands for a windshield ID (class).
    Returns:
        A dictionary of two keys ('train', 'val') accommodating the datasets.
    """    
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize(input_size),
            transforms.RandomRotation(3),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
        'val': transforms.Compose([
            transforms.Resize(input_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
    }    
    print("\nInitializing Datasets ...")
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['train', 'val']}
    print("Dataset is Loaded. \n")
    return(image_datasets)

   
def dataloader(image_datasets, P, K, val_batch_size):
    """ 
    Loads the dataloaders to iterate on for each of the training and validation datasets.
    Args:
        image_datasets (dictionary): Dictionary accomodating the two datasets (objects).
        batch_size (integer): Number of images to yield per batch for the validation set.
        P, K (integer): Number of identities and samples respectively.
    Returns:
        A dictionary of two keys ('train', 'val') accommodating the dataloaders.
    """    
    dataloaders_dict = {x:None for x in ['train', 'val']}
    print("\nInitializing Sampler and dataloaders_dict...")
    sampler = BatchSampler(image_datasets['train'], P, K)
    dataloaders_dict['val'] =  torch.utils.data.DataLoader(image_datasets['val'], batch_size = val_batch_size, shuffle = False, num_workers = 4)
    dataloaders_dict['train']  = torch.utils.data.DataLoader(image_datasets['train'], batch_sampler = sampler, num_workers = 4, pin_memory = True)
    print("Sampler and dataloaders_dict are Loaded. \n")
    return(dataloaders_dict)


    
def train_model(model, dataloaders, criterion, optimizer, num_epochs, scheduler, string_name, device):
    
    logger = SummaryWriter()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_rank1_v, best_rank5_v = 0.0, 0.0
    feats_val, labels_val = torch.ones(len(dataloaders['val'].dataset), 1024), torch.ones(len(dataloaders['val'].dataset))
    
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
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
                # for each batch update
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
                epoch_loss = running_loss/ nr_batch_triplets                
                print('{} Triplet Loss: {:.4f} Informative triplets: {:.4f}'.format(phase, epoch_loss, round(nr_batch_triplets/len(dataloaders[phase]))))
            else:
                rank1_v, rank5_v = compute_rank(feats_val, labels_val, dataloaders[phase], device)
                print('{} Rank-1: {:.4f} Rank-5: {:.4f}'.format(phase, rank1_v, rank5_v))
                scheduler.step()

            # deep copy the model
            if phase == 'val' and rank1_v > best_rank1_v:
                best_rank1_v, best_rank5_v = rank1_v, rank5_v            
                best_model_wts = copy.deepcopy(model.state_dict())
                torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict()}, string_name + '.pt')
             
        logger.add_scalars(string_name, {'train_loss':epoch_loss,
                                         'nr_triplets':nr_batch_triplets,
                                         'rank1_val': rank1_v,
                                         'rank5_val': rank5_v}, epoch+1)
    logger.close()         
    print('Training Finished. Best Validation Rank-1: {:4f} and Best val Rank-5: {:4f} '.format(best_rank1_v, best_rank5_v))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


def main(string_name, nr_epochs):

    #data_dir = '/home/efklidis/data/reranking_data/'
    data_dir = '/home/efklidis/Desktop/windshield_training/windshield_toy'

    print("Initializing dataset...")
    image_datasets  = dataset(data_dir)    
    print("Dataset is loaded.")
    
    print("\nInitializing dataloaders...")
    dataloaders = dataloader(image_datasets, P=17, K=8, val_batch_size=140)
    print("Dataloaders are booted.")

    print("\nInitializing model...")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = models.densenet121(pretrained=True).to(device)    
    print("Model is mounted on device:", str(device))

    print("\nInitializing optimizer...")  
    criterion = nn.TripletMarginLoss(margin=0.5, reduction='mean')
    optimizer = optim.Adam(model.parameters(), lr=0.005, weight_decay=0.0005) #initially at lr=0.0003
    scheduler =  optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50, 80, 110, 140, 160, 180, 200, 220], gamma=0.9)
    print("Set.")

    print("\nTraining for ", str(nr_epochs), "epochs...")
    model = train_model(model, dataloaders, criterion, optimizer, nr_epochs, scheduler, string_name, device)
    

if __name__ == '__main__':
    triplet_selector = BatchAll(0.5)
    main(string_name = 'fine_batchall', nr_epochs = 240)




