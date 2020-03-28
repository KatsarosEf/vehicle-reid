# vehicle-reid

### Github repo for the paper "A Triplet-learnt Coarse-to-Fine Reranking for Vehicle Re-identification"

#### Weights of models discussed in the paper to apply the reranking and reproduce results.<br/>
Downloadable here: https://1drv.ms/u/s!AoEekc0Cw3zhgbA0o3HSjFyaKJSg4Q?e=Oa6gik as dictionaries.<br /> 
-- coarse_model.pt<br/>
-- fine_model.pt<br/>
-- viewpoint_classifier.pt<br/>
-- detector.pt<br/> 

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
Assumes a directory where windshield images are split into train and val. Each vehicle's images are then found on unique identity folders:

/windshields/train/id_1223/front/img_1 refers to a frontal windshield image of the vehicle identity '1223' in the training set.
/windshields/train/id_1223/back/img_3 belongs to the same vehicle but is a backside windshield image.
/windshields/val/id_7536/back/img_2 refers to a frontal windshield image of the vehicle identity '7536' in the validation set.

