#### <a name='reqs'> Requirements</a>:

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

### Usage 
Clone the repository 

Setup conda environment 
 
``` markdown
conda env create -f environment.yml
conda activate dips
```

## Dataset preparation 
Please refer [here](https://github.com/clovaai/wsolevaluation) for dataset preparation 

Also refer [here](https://github.com/clovaai/wsolevaluation) for metadata preparation 

#### <a name="datasets"> Download datasets </a>:
See [folds/wsol-done-right-splits/dataset-scripts](
folds/wsol-done-right-splits/dataset-scripts). For more details, see
[wsol-done-right](https://github.com/clovaai/wsolevaluation) repo.


##### Edit the config files under configs folder
###### 1. Add paths to ImageNet dataset 
``` python
--data_root=\PATH\TO\DATASET
--metadata_root=\PATH\TO\GROUND_TRUTH 
```
