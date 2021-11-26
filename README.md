# MNIST_CLASSIFICATION
MNIST Classification - Improvements step by step

## 1. Create a new environment and activate
```
conda create --prefix ./env python=3.7 -y
conda activate /env
```

## 2. Install dependencies
```
pip install -r requirements.txt
```
## 3. Using mlflow
### 3.1 To start mlflow server
```
mlflow server \
--backend-store-uri sqlite:///mlflow.db \
--default-artifact-root ./artifacts \
--host 127.0.0.1 -p 1234
```

### 3.2 To log experiments add following lines to code
```
mlflow.set_tracking_uri("http://127.0.0.1:1234")
mlflow.set_experiment("MLFlow_TF_Autolog")

# Optionally when running with different model setup
with mlflow.start_run(run_name=f"{in_stage}") as run:
```
