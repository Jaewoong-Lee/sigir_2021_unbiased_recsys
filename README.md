# sigir_2021_unbiased_recsys
We modified the code below.
https://github.com/usaito/unbiased-implicit-rec-real

Datasets
Coat: https://www.cs.cornell.edu/~schnabts/mnar/
Yahoo! R3: https://webscope.sandbox.yahoo.com/catalog.php?datatype=r


# Dependencies
numpy==1.16.2
pandas==0.24.2
scikit-learn==0.20.3
tensorflow==1.15.0 (cpu)
plotly==3.10.0
mlflow==1.4.0
pyyaml==5.1

# Preparing
Please go to the "./src" to run this code.
(To experiment with the yahoo dataset, you need to copy train.txt and test.txt to ./data/yahoo)

# For Coat dataset
## MF
python main.py -m mf --data coat -lr 0.005 -reg 1e-9 -ran 10 -hidden 128

## Rel-MF
python main.py -m rel-mf --data coat -lr 0.005 -reg 1e-5 -ran 10 -hidden 128

## MF-DU
python main.py -m mf-du --data coat -lr 0.01 -reg 1e-7 -ran 10 -hidden 128


# For Yahoo! R3 dataset
## MF
python main.py -m mf --data yahoo -lr 0.001 -reg 1e-7 -ran 5 -hidden 64

## Rel-MF
python main.py -m rel-mf --data yahoo -lr 0.001 -reg 1e-7 -ran 5 -hidden 64

## MF-DU
python main.py -m mf-du --data yahoo -lr 0.001 -reg 1e-8 -ran 5 -hidden 64


# Results
You can see the experimental results in the "./logs".
