# RESULTS

## Q1(A) 
### MNIST Dataset

| Dataset | Epochs | Batch Size | Optimizer | Learning Rate | Model    | Test Accuracy (%) | Best Val Acc (%) | pin_memory |
|---------|--------|------------|-----------|---------------|----------|-------------------|------------------|------------|
| MNIST   | 3      | 16         | SGD       | 0.001         | RESNET18 | 99.06             | 99.16            | FALSE      |
| MNIST   | 3      | 16         | SGD       | 0.001         | RESNET50 | 98.76             | 98.83            | FALSE      |
| MNIST   | 3      | 16         | SGD       | 0.001         | RESNET18 | 98.9              | 98.94            | TRUE       |
| MNIST   | 3      | 16         | SGD       | 0.001         | RESNET50 | 98.76             | 98.97            | TRUE       |
| MNIST   | 3      | 16         | SGD       | 0.0001        | RESNET18 | 96.58             | 96.79            | FALSE      |
| MNIST   | 3      | 16         | SGD       | 0.0001        | RESNET50 | 95.84             | 96.01            | FALSE      |
| MNIST   | 3      | 16         | SGD       | 0.0001        | RESNET18 | 97.13             | 97.03            | TRUE       |
| MNIST   | 3      | 16         | SGD       | 0.0001        | RESNET50 | 96.16             | 96.56            | TRUE       |
| MNIST   | 3      | 16         | Adam      | 0.001         | RESNET18 | 98.31             | 98.51            | FALSE      |
| MNIST   | 3      | 16         | Adam      | 0.001         | RESNET50 | 98.51             | 98.7             | FALSE      |
| MNIST   | 3      | 16         | Adam      | 0.001         | RESNET18 | 98.36             | 98.96            | TRUE       |
| MNIST   | 3      | 16         | Adam      | 0.001         | RESNET50 | 98.39             | 98.79            | TRUE       |
| MNIST   | 3      | 16         | Adam      | 0.0001        | RESNET18 | 99.11             | 99.3             | FALSE      |
| MNIST   | 3      | 16         | Adam      | 0.0001        | RESNET50 | 98.54             | 98.74            | FALSE      |
| MNIST   | 3      | 16         | Adam      | 0.0001        | RESNET18 | 98.69             | 99.14            | TRUE       |
| MNIST   | 3      | 16         | Adam      | 0.0001        | RESNET50 | 98.38             | 98.74            | TRUE       |
| MNIST   | 3      | 32         | SGD       | 0.001         | RESNET18 | 98.65             | 98.71            | FALSE      |
| MNIST   | 3      | 32         | SGD       | 0.001         | RESNET50 | 98.35             | 98.56            | FALSE      |
| MNIST   | 3      | 32         | SGD       | 0.001         | RESNET18 | 98.91             | 99.06            | TRUE       |
| MNIST   | 3      | 32         | SGD       | 0.001         | RESNET50 | 98.11             | 98.46            | TRUE       |
| MNIST   | 3      | 32         | SGD       | 0.0001        | RESNET18 | 95.01             | 95.61            | FALSE      |
| MNIST   | 3      | 32         | SGD       | 0.0001        | RESNET50 | 85.47             | 85.53            | FALSE      |
| MNIST   | 3      | 32         | SGD       | 0.0001        | RESNET18 | 94.69             | 95.31            | TRUE       |
| MNIST   | 3      | 32         | SGD       | 0.0001        | RESNET50 | 86.38             | 87.07            | TRUE       |
| MNIST   | 3      | 32         | Adam      | 0.001         | RESNET18 | 98.49             | 98.64            | FALSE      |
| MNIST   | 3      | 32         | Adam      | 0.001         | RESNET50 | 97.24             | 98.26            | FALSE      |
| MNIST   | 3      | 32         | Adam      | 0.001         | RESNET18 | 98.66             | 98.91            | TRUE       |
| MNIST   | 3      | 32         | Adam      | 0.001         | RESNET50 | 96.47             | 98.33            | TRUE       |
| MNIST   | 3      | 32         | Adam      | 0.0001        | RESNET18 | 98.64             | 99.07            | FALSE      |
| MNIST   | 3      | 32         | Adam      | 0.0001        | RESNET50 | 98.85             | 98.9             | FALSE      |
| MNIST   | 3      | 32         | Adam      | 0.0001        | RESNET18 | 98.86             | 99.09            | TRUE       |
| MNIST   | 3      | 32         | Adam      | 0.0001        | RESNET50 | 98.67             | 98.76            | TRUE       |
| MNIST   | 5      | 16         | SGD       | 0.001         | RESNET18 | 99.13             | 99.34            | FALSE      |
| MNIST   | 5      | 16         | SGD       | 0.001         | RESNET50 | 99.04             | 99.17            | FALSE      |
| MNIST   | 5      | 16         | SGD       | 0.001         | RESNET18 | 99.18             | 99.36            | TRUE       |
| MNIST   | 5      | 16         | SGD       | 0.001         | RESNET50 | 98.97             | 99.21            | TRUE       |
| MNIST   | 5      | 16         | SGD       | 0.0001        | RESNET18 | 98.06             | 98               | FALSE      |
| MNIST   | 5      | 16         | SGD       | 0.0001        | RESNET50 | 97.24             | 97.56            | FALSE      |
| MNIST   | 5      | 16         | SGD       | 0.0001        | RESNET18 | 97.71             | 98.06            | TRUE       |
| MNIST   | 5      | 16         | SGD       | 0.0001        | RESNET50 | 96.47             | 96.87            | TRUE       |
| MNIST   | 5      | 16         | Adam      | 0.001         | RESNET18 | 98.84             | 99.23            | FALSE      |
| MNIST   | 5      | 16         | Adam      | 0.001         | RESNET50 | 98.64             | 98.83            | FALSE      |
| MNIST   | 5      | 16         | Adam      | 0.001         | RESNET18 | 99.16             | 99.2             | TRUE       |
| MNIST   | 5      | 16         | Adam      | 0.001         | RESNET50 | 98.56             | 99.17            | TRUE       |
| MNIST   | 5      | 16         | Adam      | 0.0001        | RESNET18 | 99.21             | 99.43            | FALSE      |
| MNIST   | 5      | 16         | Adam      | 0.0001        | RESNET50 | 98.88             | 99.19            | FALSE      |
| MNIST   | 5      | 16         | Adam      | 0.0001        | RESNET18 | 99.09             | 99.37            | TRUE       |
| MNIST   | 5      | 16         | Adam      | 0.0001        | RESNET50 | 98.81             | 98.96            | TRUE       |
| MNIST   | 5      | 32         | SGD       | 0.001         | RESNET18 | 99.01             | 99.2             | FALSE      |
| MNIST   | 5      | 32         | SGD       | 0.001         | RESNET50 | 98.48             | 98.76            | FALSE      |
| MNIST   | 5      | 32         | SGD       | 0.001         | RESNET18 | 98.88             | 99.13            | TRUE       |
| MNIST   | 5      | 32         | SGD       | 0.001         | RESNET50 | 98.73             | 98.94            | TRUE       |
| MNIST   | 5      | 32         | SGD       | 0.0001        | RESNET18 | 96.59             | 96.8             | FALSE      |
| MNIST   | 5      | 32         | SGD       | 0.0001        | RESNET50 | 94.56             | 95.24            | FALSE      |
| MNIST   | 5      | 32         | SGD       | 0.0001        | RESNET18 | 96.62             | 97.1             | TRUE       |
| MNIST   | 5      | 32         | SGD       | 0.0001        | RESNET50 | 94.81             | 95.47            | TRUE       |
| MNIST   | 5      | 32         | Adam      | 0.001         | RESNET18 | 98.7              | 99               | FALSE      |
| MNIST   | 5      | 32         | Adam      | 0.001         | RESNET50 | 98.31             | 98.94            | FALSE      |
| MNIST   | 5      | 32         | Adam      | 0.001         | RESNET18 | 98.96             | 99.09            | TRUE       |
| MNIST   | 5      | 32         | Adam      | 0.001         | RESNET50 | 98.06             | 98.89            | TRUE       |
| MNIST   | 5      | 32         | Adam      | 0.0001        | RESNET18 | 99.04             | 99.2             | FALSE      |
| MNIST   | 5      | 32         | Adam      | 0.0001        | RESNET50 | 98.36             | 98.87            | FALSE      |
| MNIST   | 5      | 32         | Adam      | 0.0001        | RESNET18 | 99.29             | 99.4             | TRUE       |
| MNIST   | 5      | 32         | Adam      | 0.0001        | RESNET50 | 98.14             | 98.93            | TRUE       |

### FashionMNIST Dataset

| Dataset      | Epochs | Batch Size | Optimizer | Learning Rate | Model    | Test Accuracy (%) | Best Val Acc (%) | pin_memory |
|--------------|--------|------------|-----------|---------------|----------|-------------------|------------------|------------|
| FashionMNIST | 3      | 16         | SGD       | 0.001         | RESNET18 | 91.37             | 91.46            | FALSE      |
| FashionMNIST | 3      | 16         | SGD       | 0.001         | RESNET50 | 89.19             | 88.96            | FALSE      |
| FashionMNIST | 3      | 16         | SGD       | 0.001         | RESNET18 | 90.8              | 90.41            | TRUE       |
| FashionMNIST | 3      | 16         | SGD       | 0.001         | RESNET50 | 89.87             | 89.64            | TRUE       |
| FashionMNIST | 3      | 16         | SGD       | 0.0001        | RESNET18 | 84.91             | 84.89            | FALSE      |
| FashionMNIST | 3      | 16         | SGD       | 0.0001        | RESNET50 | 81.54             | 81.4             | FALSE      |
| FashionMNIST | 3      | 16         | SGD       | 0.0001        | RESNET18 | 84.43             | 84.83            | TRUE       |
| FashionMNIST | 3      | 16         | SGD       | 0.0001        | RESNET50 | 83.43             | 83.44            | TRUE       |
| FashionMNIST | 3      | 16         | Adam      | 0.001         | RESNET18 | 91.5              | 91.34            | FALSE      |
| FashionMNIST | 3      | 16         | Adam      | 0.001         | RESNET50 | 90.79             | 91.04            | FALSE      |
| FashionMNIST | 3      | 16         | Adam      | 0.001         | RESNET18 | 91.95             | 91.59            | TRUE       |
| FashionMNIST | 3      | 16         | Adam      | 0.001         | RESNET50 | 89.73             | 90               | TRUE       |
| FashionMNIST | 3      | 16         | Adam      | 0.0001        | RESNET18 | 91.91             | 91.39            | FALSE      |
| FashionMNIST | 3      | 16         | Adam      | 0.0001        | RESNET50 | 89.99             | 89.77            | FALSE      |
| FashionMNIST | 3      | 16         | Adam      | 0.0001        | RESNET18 | 91.96             | 91.7             | TRUE       |
| FashionMNIST | 3      | 16         | Adam      | 0.0001        | RESNET50 | 91.79             | 91.51            | TRUE       |
| FashionMNIST | 3      | 32         | SGD       | 0.001         | RESNET18 | 89.55             | 89.47            | FALSE      |
| FashionMNIST | 3      | 32         | SGD       | 0.001         | RESNET50 | 88.21             | 88.21            | FALSE      |
| FashionMNIST | 3      | 32         | SGD       | 0.001         | RESNET18 | 89.66             | 89.7             | TRUE       |
| FashionMNIST | 3      | 32         | SGD       | 0.001         | RESNET50 | 87.49             | 87.43            | TRUE       |
| FashionMNIST | 3      | 32         | SGD       | 0.0001        | RESNET18 | 82.61             | 82.7             | FALSE      |
| FashionMNIST | 3      | 32         | SGD       | 0.0001        | RESNET50 | 81.4              | 82.04            | FALSE      |
| FashionMNIST | 3      | 32         | SGD       | 0.0001        | RESNET18 | 81.28             | 83.73            | TRUE       |
| FashionMNIST | 3      | 32         | SGD       | 0.0001        | RESNET50 | 80.56             | 81.71            | TRUE       |
| FashionMNIST | 3      | 32         | Adam      | 0.001         | RESNET18 | 91.9              | 91.91            | FALSE      |
| FashionMNIST | 3      | 32         | Adam      | 0.001         | RESNET50 | 90.11             | 90.31            | FALSE      |
| FashionMNIST | 3      | 32         | Adam      | 0.001         | RESNET18 | 90.99             | 90.84            | TRUE       |
| FashionMNIST | 3      | 32         | Adam      | 0.001         | RESNET50 | 89.14             | 88.54            | TRUE       |
| FashionMNIST | 3      | 32         | Adam      | 0.0001        | RESNET18 | 91.19             | 91.03            | FALSE      |
| FashionMNIST | 3      | 32         | Adam      | 0.0001        | RESNET50 | 90.32             | 90.21            | FALSE      |
| FashionMNIST | 3      | 32         | Adam      | 0.0001        | RESNET18 | 91.83             | 91.67            | TRUE       |
| FashionMNIST | 3      | 32         | Adam      | 0.0001        | RESNET50 | 90.84             | 90.23            | TRUE       |
| FashionMNIST | 5      | 16         | SGD       | 0.001         | RESNET18 | 90.59             | 91.49            | FALSE      |
| FashionMNIST | 5      | 16         | SGD       | 0.001         | RESNET50 | 91.77             | 91.71            | FALSE      |
| FashionMNIST | 5      | 16         | SGD       | 0.001         | RESNET18 | 92.23             | 92.09            | TRUE       |
| FashionMNIST | 5      | 16         | SGD       | 0.001         | RESNET50 | 90.68             | 91.36            | TRUE       |
| FashionMNIST | 5      | 16         | SGD       | 0.0001        | RESNET18 | 87.92             | 87.94            | FALSE      |
| FashionMNIST | 5      | 16         | SGD       | 0.0001        | RESNET50 | 83.48             | 83.91            | FALSE      |
| FashionMNIST | 5      | 16         | SGD       | 0.0001        | RESNET18 | 87.45             | 87.5             | TRUE       |
| FashionMNIST | 5      | 16         | SGD       | 0.0001        | RESNET50 | 83.65             | 83.7             | TRUE       |
| FashionMNIST | 5      | 16         | Adam      | 0.001         | RESNET18 | 92.42             | 92.44            | FALSE      |
| FashionMNIST | 5      | 16         | Adam      | 0.001         | RESNET50 | 86.56             | 90.04            | FALSE      |
| FashionMNIST | 5      | 16         | Adam      | 0.001         | RESNET18 | 92.43             | 92.46            | TRUE       |
| FashionMNIST | 5      | 16         | Adam      | 0.001         | RESNET50 | 91.94             | 91.99            | TRUE       |
| FashionMNIST | 5      | 16         | Adam      | 0.0001        | RESNET18 | 91.72             | 92.1             | FALSE      |
| FashionMNIST | 5      | 16         | Adam      | 0.0001        | RESNET50 | 92.46             | 92.51            | FALSE      |
| FashionMNIST | 5      | 16         | Adam      | 0.0001        | RESNET18 | 92.81             | 93.03            | TRUE       |
| FashionMNIST | 5      | 16         | Adam      | 0.0001        | RESNET50 | 92.26             | 91.86            | TRUE       |
| FashionMNIST | 5      | 32         | SGD       | 0.001         | RESNET18 | 91.34             | 91.27            | FALSE      |
| FashionMNIST | 5      | 32         | SGD       | 0.001         | RESNET50 | 87.79             | 90.57            | FALSE      |
| FashionMNIST | 5      | 32         | SGD       | 0.001         | RESNET18 | 90.59             | 91.03            | TRUE       |
| FashionMNIST | 5      | 32         | SGD       | 0.001         | RESNET50 | 89.95             | 90.47            | TRUE       |
| FashionMNIST | 5      | 32         | SGD       | 0.0001        | RESNET18 | 84.33             | 84.79            | FALSE      |
| FashionMNIST | 5      | 32         | SGD       | 0.0001        | RESNET50 | 83.14             | 83.99            | FALSE      |
| FashionMNIST | 5      | 32         | SGD       | 0.0001        | RESNET18 | 84.04             | 84.24            | TRUE       |
| FashionMNIST | 5      | 32         | SGD       | 0.0001        | RESNET50 | 82.79             | 82.61            | TRUE       |
| FashionMNIST | 5      | 32         | Adam      | 0.001         | RESNET18 | 92.35             | 92.2             | FALSE      |
| FashionMNIST | 5      | 32         | Adam      | 0.001         | RESNET50 | 91.12             | 91.23            | FALSE      |
| FashionMNIST | 5      | 32         | Adam      | 0.001         | RESNET18 | 92.13             | 92.24            | TRUE       |
| FashionMNIST | 5      | 32         | Adam      | 0.001         | RESNET50 | 90.8              | 90.96            | TRUE       |
| FashionMNIST | 5      | 32         | Adam      | 0.0001        | RESNET18 | 92.59             | 92.17            | FALSE      |
| FashionMNIST | 5      | 32         | Adam      | 0.0001        | RESNET50 | 91.59             | 91.74            | FALSE      |
| FashionMNIST | 5      | 32         | Adam      | 0.0001        | RESNET18 | 91.84             | 92.09            | TRUE       |
| FashionMNIST | 5      | 32         | Adam      | 0.0001        | RESNET50 | 91.42             | 91.57            | TRUE       |

## Q1(B)
### MNIST Dataset

| Dataset | Kernel | Test Accuracy (%) | Train Time (ms) |
|---------|--------|-------------------|-----------------|
| MNIST   | poly   | 86.88             | 21563.77        |
| MNIST   | rbf    | 92.18             | 10941.59        |

### FashionMNIST Dataset

| Dataset      | Kernel | Test Accuracy (%) | Train Time (ms) |
|--------------|--------|-------------------|-----------------|
| FashionMNIST | poly   | 82.52             | 11123.27        |
| FashionMNIST | rbf    | 85.7              | 8986.36         |

## Q2
### FashionMNIST Dataset

| Compute | Batch Size | Optimizer | Learning Rate | Model    | Test Accuracy (%) | Train Time (ms) | FLOPs |
|---------|------------|-----------|---------------|----------|-------------------|-----------------|-------|
| CUDA    | 16         | Adam      | 0.001         | RESNET18 | 90.81             | 281759.92       | 1.82G |
| CUDA    | 16         | Adam      | 0.001         | RESNET50 | 89.89             | 504351.68       | 4.13G |
| CUDA    | 16         | SGD       | 0.001         | RESNET18 | 89.71             | 270698.06       | 1.82G |
| CUDA    | 16         | SGD       | 0.001         | RESNET50 | 87.54             | 479728.41       | 4.13G |
| CPU     | 16         | Adam      | 0.001         | RESNET18 | 90.6              | 30013417.85     | 1.82G |
| CPU     | 16         | Adam      | 0.001         | RESNET50 | 85.3              | 44885390        | 4.13G |
| CPU     | 16         | SGD       | 0.001         | RESNET18 | 90.21             | 29142641.76     | 1.82G |
| CPU     | 16         | SGD       | 0.001         | RESNET50 | 83.25             | 42587392        | 4.13G |
