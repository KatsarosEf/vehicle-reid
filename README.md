# vehicle-reid

### Github repo for the paper "A Triplet-learnt Coarse-to-Fine Reranking for Vehicle Re-identification"

#### Weights of models discussed in the paper to apply the reranking and reproduce results.<br/>
Downloadable here: https://1drv.ms/u/s!AoEekc0Cw3zhgbA0o3HSjFyaKJSg4Q?e=Oa6gik as dictionaries.<br /> 
- ###### coarse_model.pt (224,224,3) (vehicle image) ---> (1024) (triplet-learnt coarse embedding) <br/>
- ###### fine_model.pt (62,152,3) (windshield image) ---> (1024) (triplet-learnt fine embedding) <br/>
- ###### viewpoint_classifier.pt (1024) (coarse embedding) ---> (2) (class probabilities) <br/>
- ###### detector.pt (224,224,3) (vehicle image) ---> (4) (box coordinates) & 2 (class probabilities) <br/> 

#### Train the models discussed in the paper from scratch.<br/>
###### train_coarse.py 
It is used to train the coarse densenet121 model utilizing the triplet loss and BatchAll sampling technique. The coarse triplet model projects similar vehicle images closer together and disimilar ones further apart.
Assumes a directory where images are split into train and val. Each vehicle's images are then found on unique identity folders.

/vehicles/train/id_1223/img_7.jpg refers the seventh image of the vehicle identity '1223' in the training set.
/vehicles/val/id_7536/img_2.jpg refers the second image of the vehicle identity '7536' that has been allocated for validation. 

###### train_classifier.py <br /> 
It is used to train the viewpoint classification model. Requires the trained coarse model to evaluate triplet embeddings before inputting them for classification. The viewpoint classifier takes an input image and determines whether it is the frontal or backside vehicle viewpoint.
Assumes a directory where images are split into train, val and optionally test sets:

 /viewpoints/train/front/img_108.jpg refers to an image of the training set. It is captured from the frontal viewpoint.
 /viewpoints/val/back/img_321.jpg refers to an image of the validation set. It is captured from the backside point of view.
 
###### train_fine.py <br /> 
It is used to train the fine model. The fine triplet model takes windshield images and - similarly to the coarse - projects similar ones closer together and disimilar ones further apart.
Assumes a directory where windshield images are firstly classifier as frontal or backside-viewed w.r.t. the viewpoint classifier. Likewise, we construct the windshield classes (identities):

/windshields/train/id_1223_front/img_1.jpg refers to a frontal windshield image of the vehicle identity '1223' in the training set.
/windshields/train/id_1223_back/img_3.jpg belongs to the same vehicle but is a backside windshield image.
/windshields/val/id_7536_back/img_2.jpg refers to a frontal windshield image of the vehicle identity '7536' in the validation set.

###### train_detector.py <br /> 
It is employed to fine-tune the detector that comes pre-trained on the COCO dataset. We change the classification nodes from 81 (COCO) to 2 (windshield vs background). We use 700 annotated images (bounding boxes), 500 for training and 200 for validation. It assumes a directory of 2 subdirs ('images' and 'annotations') and 2 text files indicating which images are selected for training and which ones for validation:

/detection/images/... contains the raw images. <br />
/detection/annotations/... contains the .xml annotations for each raw image. <br />
/detection/train.txt contains the names of images allocated for training. <br />
/detection/val.txt contains the names of images assigned on the validation set. <br />


