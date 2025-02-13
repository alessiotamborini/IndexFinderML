# IndexFinderML

This library contains the work for the machine learning-based automated cardiac pressure waveform fiducial point detection.

## Table of Contents
1. [Introduction](#introduction)
2. [Installation](#installation)
3. [Usage](#usage)
4. [Features](#features)
5. [Contributing](#contributing)
6. [License](#license)
7. [Contact](#contact)

## Introduction
Provide a brief overview of the project, its purpose, and its goals.

![Cardiac Pressure Waveform](./assets/model_diagram.png)

Make sure to replace `path/to/your/image.png` with the actual path to your image file.

## Installation

To install and set up the project, follow these steps:

1. **Clone the repository**:
    ```bash
    git clone https://github.com/yourusername/IndexFinderML.git
    cd IndexFinderML
    ```

2. **Create a virtual environment** (optional but recommended):
    ```bash
    python3 -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. **Install the required dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

4. **Verify the installation**:
    ```bash
    python -m unittest discover tests
    ```

You should now have the project set up and ready to use.

## Usage
Examples and explanations on how to use the project.

## Model Checkpoints

Two model versions of the model are available for use:

1. **Model Checkpoint with Augmented Data**: This version of the model has been trained using augmented data, which can help improve the model's robustness and performance on a wider range of inputs.

2. **Model Checkpoint without Augmented Data**: This version of the model has been trained without any data augmentation, providing a baseline performance on the original dataset.

## Features
List of features included in the project.

## Contributing
Guidelines for contributing to the project.

## License
Information about the project's license.

## Contact
Contact information for the project maintainers.
