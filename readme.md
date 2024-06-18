# smart_contract_analyser

## Overview

Voltius Core is a project for smart contract verification in Solidity using a Graph Neural Network (GNN). This project includes data preprocessing, feature extraction, model training, evaluation, and prediction.

## Requirements

Install the required packages using the following command:

```bash
pip install -r requirements.txt
```

Directory Structure

smart_contract_analysis/ - Contains scripts for graph neural network training and prediction.
dataset_contracts/ - Directory with graph data for smart contracts.

Usage
Training the Model

To train the model, run:

```bash
python smart_contract_analysis/gnn_model.py
```

Predicting Vulnerabilities

To predict vulnerabilities in a smart contract, run:
```bash
python smart_contract_analysis/predict_vulnerability.py
```

Authors

    Voltius team
    Contributors

License

This project is licensed under the MIT License - see the LICENSE file for details.
