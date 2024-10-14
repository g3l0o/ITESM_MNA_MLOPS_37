# ITESM_MNA_MLOPS_37

## Miembros del equipo
* Carlos Mariano Ramírez Amaya 	~> A01795036
* Rogelio Rivera Meléndez	~> A01166618
* Felipe Enrique Vázquez Ruiz	~> A01638116
* Andrea Fernanda Molina Blandón	~> A00827133
* Jorge Olavarrieta de la Torre	~> A01795487
* Edgar Gerardo Rojas Medina	~> A00840712


## Project Structure

```bash
.
├── README.md
├── data
│   ├── final
│   │   ├── test.csv
│   │   └── train.csv
│   ├── model
│   │   └── cc.en.300.bin
│   ├── pre_processed
│   │   └── pricerunner.csv
│   ├── processed
│   │   └── pricerunner.csv
│   ├── raw
│   │   ├── pricerunner_aggregate.csv
│   │   └── pricerunner_aggregate.csv.dvc
│   └── transformed
│       ├── pricerunner_test_features.csv
│       ├── pricerunner_test_target.csv
│       ├── pricerunner_train_features.csv
│       └── pricerunner_train_target.csv
├── docs
├── dvc.yaml
├── refactoring
│   ├── __init__.py
│   └── v1
│       ├── __init__.py
│       ├── data_loader.py
│       ├── pipelineObj.py
│       ├── preproccesing.py
│       ├── requirements.txt
│       └── transformation.py
├── requirements.txt
└── src
    ├── __init__.py
    ├── data_loader.py
    ├── embedder.py
    ├── evaluator.py
    ├── model.py
    ├── model_downloader.py
    ├── processor.py
    ├── transformer.py
    └── utils.py
```

## Set up the environment


1. Create the virtual environment:
```bash
python3 -m venv mlops_env
```
2. Activate the virtual environment:

- For Linux/MacOS:
```bash
source mlops_env/bin/activate
```
- For Command Prompt:
```bash
.\mlops_env\Scripts\activate
```
3. Install dependencies:

- To install only production dependencies, run:
```bash
pip install -r requirements.txt
```
- To install a new package, run:
```bash
pip install <package-name>
```


## DVC

1. Initialize dvc
```bash
dvc init
```

2. Add the dataset to DVC tracking

```bash
dvc add pricerunner_aggregate.csv
```

3. Create DVC pipeline

```bash
dvc stage add -n load_data \
-d data/raw/pricerunner_aggregate.csv \
-o data/pre_processed/pricerunner.csv \
python src/data_loader.py data/raw/pricerunner_aggregate.csv data/pre_processed/pricerunner.csv
```


