# Deep Learning HMAX Models for the Serre Lab

## Pytorch Models
1. Pytorch version of the Zelinsky DeepHMAX from: https://doi.org/10.1080/13506285.2019.1661927 
2. HMAX model based on original HMAX from: https://www.pnas.org/content/104/15/6424
- This model requires Gabor filters which can be generated from `gabor_filters.py`. An example of how to prepare the filters is given at the end of the file. 

## Running the Models
Models can be referenced in `run_imagenet.py` and run on Oscar CCV using this code. In order to run this on imagenet in other platforms, the data directory would have to be changed to where imagenet is stored. 

### Example:
```
python run_imagenet.py --epochs 90 -b 382 -j 16
```
