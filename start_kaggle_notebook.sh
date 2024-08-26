#!/bin/zsh

# Define the path to your notebook and the Kaggle kernel metadata
NOTEBOOK_PATH="path/to/your/notebook.ipynb"
KERNEL_METADATA_PATH="path/to/kernel-metadata.json"

# Push the notebook to Kaggle
kaggle kernels push -p $KERNEL_METADATA_PATH

# Start the notebook on Kaggle
kaggle kernels status $KERNEL_METADATA_PATH
