### Code for extended paper has been released that can be used using main_extended.py.
#### In addition you first need to train a linear classifier on top of the dino feature and then copy those weights in the root folder of this repository.

### Citation(s):
```
@inproceedings{murtaza2023discriminative,
  title={Discriminative sampling of proposals in self-supervised transformers for weakly supervised object localization},
  author={Murtaza, Shakeeb and Belharbi, Soufiane and Pedersoli, Marco and Sarraf, Aydin and Granger, Eric},
  booktitle={Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision Workshops},
  pages={155--165},
  year={2023}
}
```
```
@article{MURTAZA2023dips,
title = {DiPS: Discriminative Pseudo-label sampling with self-supervised transformers for weakly supervised object localization},
author = {Shakeeb Murtaza and Soufiane Belharbi and Marco Pedersoli and Aydin Sarraf and Eric Granger},
journal = {Image and Vision Computing},
pages = {104838},
year = {2023},
issn = {0262-8856},
doi = {https://doi.org/10.1016/j.imavis.2023.104838},
url = {https://www.sciencedirect.com/science/article/pii/S0262885623002123},
}
```

### Pytorch code for: Discriminative Sampling of Proposals in Self-Supervised Transformers for Weakly Supervised Object Localization

#### 1. Requirements:

* Python 3.9.7
* [Pytorch](https://github.com/pytorch/pytorch)  1.10.0
* [torchvision](https://github.com/pytorch/vision) 0.11.1
* [Full dependencies](environment.yml)
* Build and install CRF:
    * Install [Swig](http://www.swig.org/index.php)
    * CRF
```shell
cd dlib/crf/crfwrapper/bilateralfilter
swig -python -c++ bilateralfilter.i
python setup.py install
```

#### 2. Usage 
##### ○ Clone the repository 
##### ○ Setup conda environment using:
``` markdown
conda env create -f environment.yml
conda activate dips
```

#### 3. Dataset preparation 
Please refer [here](https://github.com/clovaai/wsolevaluation) for dataset preparation 

Also refer [here](https://github.com/clovaai/wsolevaluation) for metadata preparation 

Also refer [here](https://github.com/clovaai/wsolevaluation) for intstruction to download dataset 


#### 4. Run code

##### 4.1 To obtain region of interest for sampling foregorund/background pixels run the below given script.
###### Also set the following parameters
  ###### ○ metadata_root: Add path for metadata generated in step 3
  
  ###### ○ data_root: Add path to dataset root

``` python
extract_bbox.py 
--score_calulcation_method class_probability 
--proxy_training_set true 
--split train 
--bbox_min_size 0.75 
--evaluation_type 1_4_heads_and_mean 
--num_samples_to_save 0 
--data_root ~/datasets 
--dataset_name ILSVRC 
--metadata_root meta_data_base_path/metadata/ILSVRC 
--arch vit_small 
--patch_size 16 
--batch_size_per_gpu 32 
--num_workers 8 
--cam_curve_interval 0.5 
--search_hparameters false 
--nested_contours false 
--save_final_cams false 
--num_labels 1000 
--num_samples_to_save 0 
```
##### 4.2 We use [comet.ml](https://www.comet.com/site/) for experiment tracking. So, please add the following key in main.py file from your [comet.ml account](bbrVVBsFclbFud475m2L2WDYc).
  ###### ○ comet_workspace
  ###### ○ api_key

###### 4.3. To train DiPS run the below given scripts and set the following paramters.
###### Also set the following parameters
  ###### ○ metadata_root: Add path for metadata generated in step 3
  ###### ○ data_root: Add path to dataset root
  ###### ○ iou_regions_path: Add path to .pkl file that contains regions proposal for sampling background/foreground regions (.pkl can be generated using script in step 4.1).
  
``` python
main.py 
--data_root ~/datasets 
--metadata_root base_path_to_metadata/metadata/ILSVRC 
--iou_regions_path path_to_extracted_regions/1_4_heads_and_mean_on_train_list_imgs_with_best_head.pkl --run_for_given_setting --evaluate_checkpoint configs/ILSVRC 
--dataset_name ILSVRC 
--epochs 10
--proxy_training_set false 
--exp_pre_name train_logs_on_ILSVRC 
```

  ###### ○ train extended model

``` python
main.py 
--seed_type MBProbSeederSLCamsWithROI 
--arch vit_small 
--patch_size 16
--data_root ~/datasets 
--metadata_root base_path_to_metadata/metadata/ILSVRC 
--iou_regions_path path_to_extracted_regions/1_4_heads_and_mean_on_train_list_imgs_with_best_head.pkl --run_for_given_setting --evaluate_checkpoint configs/ILSVRC 
--dataset_name ILSVRC 
--epochs 10
--proxy_training_set false 
--exp_pre_name train_logs_on_ILSVRC 
--batch_size_per_gpu 32 
--experiment_category extended 
--warmup_epochs 0 
--num_workers 8 
--evaluation_type 1_4_heads_and_mean 
--iou_regions_path 'set_path_of_your_extracted_bbox' 
--number_of_heads_for_fusion 0 
--use_image_as_input_for_fusion true 
--cam_curve_interval 0.001 
--mask_root dataset 
--search_hparameters true 
--use_dino_classifier true 
 --classifier_for_top1_top5 Dino_Head 
```

#### 5. Credit: We borrow some code from the following repositories:

##### ○ [https://github.com/clovaai/wsolevaluation](https://github.com/clovaai/wsolevaluation)
##### ○ [https://github.com/sbelharbi/fcam-wsol](https://github.com/sbelharbi/fcam-wsol)

##### ○ [https://github.com/facebookresearch/dino](https://github.com/facebookresearch/dino)
