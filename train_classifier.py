from __future__ import print_function
from __future__ import division
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torch.nn.functional as F
from torchvision import datasets, models, transforms
import os
import copy
from torch.utils.tensorboard import SummaryWriter

print("PyTorch Version: ",torch.__version__)
print("Torchvision Version: ",torchvision.__version__)



def dataset(data_dir, input_size = 224):
    """ 
    Loads the transformations required for ImageNet-pretrained models and initializes training and validation datasets.
    Args:
        data_dir (string): Data directory. Each folder within the directory stands for a vehicle ID (class).
    Returns:
        A dictionary of two keys ('train', 'val') accommodating the datasets.
    """    
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


def train_model(model, coarse_model, dataloaders, criterion, optimizer, nr_epochs, scheduler, string_name, device):

    logger = SummaryWriter()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    coarse_model.eval()
    
    for epoch in range(nr_epochs):
        print('Epoch {}/{}'.format(epoch, nr_epochs - 1))
        print('-' * 10)       
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  
            else:
                model.eval()  
            # for each batch
            running_loss, running_corrects = 0.0, 0
            for index, (inputs, labels) in enumerate(dataloaders[phase]):
                inputs, labels = inputs.to(device), labels.to(device)
                # evaluate coarse embeddings
                out = coarse_model.features(inputs)                
                out = F.relu(out, inplace=True)
                embeddings = F.adaptive_avg_pool2d(out, (1, 1)).view(out.size(0), -1)
                # train the classifier
                outputs = model(embeddings)
                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == 'train'):
                    if phase == 'train':
                        loss = criterion(outputs, labels)
                        loss.backward()
                        optimizer.step()                        
                    _, preds = torch.max(outputs, 1)
                # batch statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            # epoch statistics
            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                torch.save(model.state_dict(), string_name + '.pt')
            if phase == 'val':
                scheduler.step()
            # update logs
            logger.add_scalars(os.path.join(string_name, phase), {'loss':epoch_loss,
                               'accuracy':epoch_acc.item()}, epoch+1)
    logger.close()         
    print('Training is finished. Best val Acc: {:4f}'.format(best_acc))
    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


def main(string_name, nr_epochs, batch_size):
        
    data_dir = '/home/efklidis/Desktop/Front_back/data/'

    print("Initializing dataset...")
    image_datasets  = dataset(data_dir)    
    print("Dataset is loaded.")
    
    print("\nInitializing dataloaders...")
    dataloaders = {'train':torch.utils.data.DataLoader(image_datasets['train'], batch_size=batch_size, shuffle=True, num_workers=4),
                    'val':torch.utils.data.DataLoader(image_datasets['val'], batch_size=batch_size, shuffle=False, num_workers=4)}
    print("Dataloaders are booted.")

    print("\nInitializing viewpoint classification model and loading the coarse trained one...")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    coarse_model = models.DenseNet()
    coarse_wts = coarse_model.state_dict()
    
    trained_dict = torch.load('/home/efklidis/Desktop/triplet_BA.pt', map_location='cpu' )
    trained_dict = {k: v for k, v in trained_dict.items() if k in coarse_wts}

    coarse_wts.update(trained_dict) 
    coarse_model.load_state_dict(coarse_wts)
    coarse_model = coarse_model.to(device)

    view_cls = torch.nn.Sequential(torch.nn.Linear(1024, 256), torch.nn.ReLU(),
                                   torch.nn.Linear(256, 2)).to(device)    
    print("Models are mounted on device:", str(device))

    print("\nInitializing optimizer...")  
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(view_cls.parameters(), lr=0.01, momentum=0.9)
    scheduler =  optim.lr_scheduler.MultiStepLR(optimizer, milestones=[40, 80, 100, 120, 140, 160, 180], gamma=0.9)
    print("Set.")

    print("\nTraining for ", str(nr_epochs), "epochs...")
    view_cls = train_model(view_cls, coarse_model, dataloaders, criterion, optimizer, nr_epochs, scheduler, string_name, device)


if __name__ == '__main__':  
    main(string_name='viewpoint_classifier', nr_epochs=200, batch_size=12)
    
    
    