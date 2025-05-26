# <div align="center">CodeMixToolkit</div>

<div align="center">

![Docker](https://img.shields.io/badge/Docker-Enabled-2496ED?style=flat&logo=docker&logoColor=white)
![Python](https://img.shields.io/badge/Python-Supported-3776AB?style=flat&logo=python&logoColor=white)
<!-- ![Languages](https://img.shields.io/badge/Languages-Hindi%2C%20English-blue) -->

</div>

## 📚 Table of Contents
- [Overview](#overview)
  - [Core Modules](#core-modules)
  - [Key Features](#key-features)
- [Installation](#installation)
  - [Development Installation](#development-installation)
- [Usage and Supported Tools](#usage-and-supported-tools)
  - [Examples](#examples)
  - [GCM Toolkit](#1-gcm-toolkit)
  - [CSNLI Tool](#2-csnli-tool)
- [Development and Contributing](#development-and-contributing)
  - [Environment Setup](#environment-setup)
  - [Development Workflow](#development-workflow)
- [License](#license)

## Overview
CodeMixToolkit is a comprehensive toolkit for processing code-mixed text, currently supporting both Hindi and English languages. It provides a suite of tools and utilities for various aspects of code-mixed text processing, from data generation to analysis.

### Core Modules

#### 1. Data Processing (`data/`)
- Utilities for handling code-mixed text data
- Data loading and ready access to datasets

#### 2. Models (`models/`)
- Implementation of various models for code-mixed text processing
- Pre-trained model interfaces for inference using the models
- Zero and Few-shot prompting of LLMs

#### 3. Training (`train/`)
- Training scripts and utilities
- Model training pipelines

#### 4. Evaluation (`evaluation/`)
- Evaluation setup for model checkpoints and LLMS

#### 5. Utilities (`utils/`)
- Common utility functions
- Text processing helpers
- Configuration management

### Key Features
- Access to code-mixed datasets
- Language identification and normalization
- Synthetic data generation capabilities
- Model training and evaluation pipelines
- Docker-based access to popular tools for ease of usage
- Comprehensive API for integration
- Easily extendable

## Installation

### Installation
You can install directly from PyPI (coming soon):

<!-- ```bash
pip install codemix
``` -->

### Development Installation
For development purposes, you can create an editable install:

```bash
# Clone the repository
git clone https://github.com/prashantkodali/CodeMixToolkit.git
cd CodeMixToolkit

# Create an editable install
pip install -e .
```

This will allow you to modify the library code and see changes immediately without reinstalling.

## Usage and Supported Tools

### Examples
Check out our [Example Notebook](examples/ExampleNotebook.ipynb) for hands-on examples of how to use the toolkit.

Additionally, there are some previously publicly released tools that we have found useful in our work. To reduce the setup overhead, we have created docker images which expose simple APIs enabling users to leverage these tools quickly.

### 1. GCM Toolkit
A powerful tool for generating synthetic code-mixed data.

- **Docker Hub**: [prakod/gcm-codemix-generator](https://hub.docker.com/r/prakod/gcm-codemix-generator)
- **Documentation**: [GCM README](gcm/README.md)

### 2. CSNLI Tool
Specialized tool for language identification and normalization of code-mixed sentences.

- **Docker Hub**: [prakod/csnli-api](https://hub.docker.com/r/prakod/csnli-api)
- **Documentation**: [CSNLI README](csnli/README.md)

## Development and Contributing

We welcome contributions to the CodeMixToolkit! Here's how you can get started:

### Environment Setup
We recommend using Conda for development. A `conda_env.yaml` file is provided for easy environment setup:

```bash
conda env create -f conda_env.yaml
conda activate codemix
```

### Development Workflow
1. Fork the repository
2. Create a new branch for your feature
3. Set up pre-commit hooks for code quality:
   ```bash
   pre-commit install
   ```
4. Make your changes following our code style:
   - [Ruff](https://github.com/astral-sh/ruff) for linting and formatting
   - [Pre-commit](https://pre-commit.com/) for pre-commit hooks
5. Run tests and ensure they pass
6. Submit a pull request

For major changes, please open an issue first to discuss what you would like to change.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

