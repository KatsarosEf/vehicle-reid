import os
import collections
import xml.etree.ElementTree as ET
from PIL import Image
import torch
import torchvision
import transforms as T
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from engine import train_one_epoch, evaluate
import math
from vision.references.detection import utils #https://github.com/pytorch/vision/blob/master/references/detection/utils.py

class WindshieldDetection(object):

    def __init__(self, root, transforms, image_set='train'):
        #root = '/home/efklidis/Desktop/detection'
        self.image_set = image_set
        self.transforms = transforms
        image_dir = os.path.join(root, 'images')
        annotation_dir = os.path.join(root, 'annotations')
        split_f = os.path.join(root, image_set.rstrip('\n') + '.txt')

        with open(os.path.join(split_f), "r") as f:
            file_names = [x.strip() for x in f.readlines()]
        self.images = [os.path.join(image_dir, x ) for x in file_names]
        self.annotations = [os.path.join(annotation_dir, x[:-4] + ".xml") for x in file_names]
        assert (len(self.images) == len(self.annotations))
        
    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is a dictionary of the XML tree.
        """
        boxes = [[math.floor(float(i)) for i in list(self.parse_voc_xml(
                ET.parse(self.annotations[index]).getroot())['annotation']['object']['bndbox'].values())]]
    
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        img = Image.open(self.images[index]).convert('RGB')
        image_id = torch.tensor([index])
        area = (boxes[:,3] - boxes[:,1]) * (boxes[:,2] - boxes[:,0])
        iscrowd= torch.zeros((1,), dtype=torch.int64)
        labels = torch.ones((1,), dtype=torch.int64)
        
        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd
        
        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.images)

    def parse_voc_xml(self, node):
        voc_dict = {}
        children = list(node)
        if children:
            def_dic = collections.defaultdict(list)
            for dc in map(self.parse_voc_xml, children):
                for ind, v in dc.items():
                    def_dic[ind].append(v)
            voc_dict = {
                node.tag:
                    {ind: v[0] if len(v) == 1 else v
                     for ind, v in def_dic.items()}
            }
        if node.text:
            text = node.text.strip()
            if not children:
                voc_dict[node.tag] = text
        return voc_dict



def get_transform():
    transforms = []
    transforms.append(T.ToTensor())
    return T.Compose(transforms)


def get_model_instance_segmentation(num_classes = 2):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True, min_size = 224)
    num_classes = 2  # 1 class (windshield) + background
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return(model)
    

def main(nr_epochs):
    # train on the GPU or on the CPU, if a GPU is not available
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    # our dataset has two classes only - background and person
    num_classes = 2
    # define datasets and  dataloaders    
    image_datasets = {x: WindshieldDetection('/home/efklidis/Desktop/detection',transforms= get_transform(), image_set=x ) for x in ['train', 'val']}
    dataloaders = {x:torch.utils.data.DataLoader(image_datasets[x], batch_size=2, shuffle=False, num_workers=4, collate_fn=utils.collate_fn) for x in ['train', 'val']}
    # get the model using our helper function
    model = get_model_instance_segmentation(num_classes).to(device)
    # construct an optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)
    # train for 10 epochs
    
    for epoch in range(nr_epochs):
        # train for one epoch, printing every 10 iterations
        train_one_epoch(model, optimizer, dataloaders['train'], device, epoch, print_freq=50)
        # update the learning rate
        lr_scheduler.step()
        # evaluate on the test dataset
        evaluate(model, dataloaders['val'], device)

    print("The windshield detector has been trained.")


if __name__ == '__main__':  
    main(nr_epochs = 10)
        