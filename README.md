# vehicle-reid

Github repo for the paper "A Triplet-learnt Coarse-to-Fine Reranking for Vehicle Re-identification".

Weights of models discussed in the paper as dictionaries.
Downloadable here: https://1drv.ms/u/s!AoEekc0Cw3zhgbA0o3HSjFyaKJSg4Q?e=Oa6gik<br /> 
-- coarse_model.pt<br/> 
-- fine_model.pt<br/>  
-- viewpoint_classifier.pt<br/> 
-- detector.pt<br/> 

-- train_coarse.py is used to train the coarse densenet121 model utilizing the triplet loss and BatchAll sampling technique. The coarse triplet model projects similar vehicle images closer together and disimilar ones further apart.
Assumes a directory where images are split into train and val. Each vehicle's images are then found on unique identity folders:

vehicles\train\id_1\img_1<br /> 
                  \img_2 <br /> 
                   ..... <br /> 
                  \img_n <br /> 
.....<br /> 
vehicles\val\id_1\img_1<br /> 
                 \img_2<br /> 
                  .....<br /> 
                 \img_n<br /> 

-- train_classifier.py is used to train the viewpoint classification model. Requires the trained coarse model to evaluate triplet embeddings before inputting them for classification. The viewpoint classifier takes an input image and determines whether it is the frontal or backside vehicle viewpoint.
Assumes a directory where images are split into train, val and optionally test sets:

viewpoints\train\img_1<br /> 
                \img_2<br /> 
                 .....<br /> 
                \img_n<br /> 
.....<br /> 
viewpoints\val\img_1<br /> 
              \img_2<br /> 
               .....<br /> 
              \img_n<br /> 

--train_fine.py is used to train the fine model. The fine triplet model takes windshield images and - similarly to the coarse - projects similar ones closer together and disimilar ones further apart.
Assumes a directory where windshield images are split into train and val. Each vehicle's images are then found on unique identity folders:

windshields\train\id_1\img_1<br /> 
                      \img_2<br /> 
                       .....<br /> 
                       \img_n<br /> 
....<br /> 
windshields\val\id_1\img_1<br /> 
                    \img_2<br /> 
                     .....<br /> 
                    \img_n<br /> 

