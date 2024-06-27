Deep Learning Models Repository

This repository contains implementations of various deep learning models created as part of coursework assignments. Each folder contains code and documentation for a specific model and task. Below is an overview of the included assignments:
Valentin Najean - 261134857
Assignment 1: Image Classification with Convolutional Neural Networks

Goal: Achieve high accuracy on the CIFAR-100 dataset, which is more complex than CIFAR-10 due to its 100 classes.
Methodology:

    Data Module:
        Load, normalize, and split data (45,000 training, 5,000 validation, and 10,000 testing images).
        Create training, validation, and test data loaders.

    Model Architectures:
        GoogLeNet:
            Implemented with Inception modules.
            Achieved limited accuracy (around 22%).
        ResNet:
            Implemented ResNet-50 and ResNet-200.
            ResNet-200 achieved better accuracy (around 55%) but required extensive training time.
            ResNet-50 on CIFAR-10 achieved 85.8% accuracy.

    Training:
        Used early stopping and checkpoint callbacks.
        Models trained with PyTorch Lightning.

Assignment 2: Text Generation with RNN

Goal: Generate text using a character-level RNN model trained on the Shakespeare dataset.
Methodology:

    Preprocessing:
        Created a vocabulary of unique characters.
        Indexed characters using char2index and index2char dictionaries.
        Implemented a ShakespeareDataset class for sequence handling.
        Set up data loaders for batching and shuffling.

    Model Architecture:
        Implemented TextGenerationLSTM class with an embedding layer and LSTM layer.
        Used a fully connected layer to predict the next character.
        Managed LSTM hidden state between batches.

    Training:
        256 embedding dimensions, 1024 LSTM hidden state dimensions.
        Trained over 20 epochs, best model at epoch 15.
        Used Adam optimizer with a learning rate of 0.0001.

    Text Generation:
        Implemented generate_text function.
        Used temperature scaling and softmax for character sampling.

Example Generated Text:

text

KING EDWARD IV:
Now here a period of tumultuous broils.
Away with Oxford to Hames Castle straight:
For Somerset, off with his guilty hand:
Madam 'ay 'em, and much better blood I begin: I pare
Before thy bond shall be such severe past
cure of the thing you wot of, unless they kept very
good diet, as I told you,--

FROTH:
All this is true.
...

Assignment 3: Multi-Task Feed Forward Neural Network for Regression and Classification

Goal: Predict house prices and classify house categories using a multi-task neural network.
Methodology:

    Feature Engineering:
        Created features like TotalSqFeet, TotalBathrooms, and AreaQuality.
        Dropped columns with data leakage and imputed missing data using KNN imputer.

    Model Architecture:
        Used embedding layers for categorical data.
        Combined embeddings with numerical data in fully connected layers.
        Split into regression and classification paths for price prediction and category classification.
        Used mean squared error for regression and cross-entropy loss for classification.

    Model Tuning:
        Utilized Optuna for hyperparameter tuning.
        Best model hyperparameters included 1 hidden layer with 512 dimensions, 14 embedding dimensions, learning rate of 0.028, Adam optimizer, Leaky ReLU activation, and 0.403 dropout rate.

    Results:
        Best validation accuracy: 72.94%
        Best validation RMSE: 75,607.17
        Test accuracy: 70.45%
        Test RMSE: 61,701.86
