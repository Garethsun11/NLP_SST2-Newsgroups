""" Multi-layer perceptron model for Assignment 1: Starter code.

You can change this code while keeping the function headers.
"""
import os
import sys
import argparse
from typing import Dict, List
import random

# import matplotlib.pyplot as plt

from util import evaluate, load_data
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

class MultilayerPerceptronModel(nn.Module):
    """ Multi-layer perceptron model for classification.
    """
    def __init__(self, num_classes, vocab_size, embedding_dim = 128):
        """ Initializes the model.
        
        Inputs:
            num_classes (int): The number of classes.
            vocab_size (int): The size of the vocabulary.
        """
        # TODO: Implement initialization of this model.
        # Note: You can add new arguments, with a default value specified.
        super(MultilayerPerceptronModel, self).__init__()
        self.embedding = nn.Embedding(num_embeddings = vocab_size + 1, embedding_dim = embedding_dim)
        self.mlp_1 = torch.nn.Sequential(
            nn.Linear(embedding_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            # nn.ReLU(),
            nn.Linear(64, num_classes),
            # nn.Softmax(dim = 1)
        )
    
    def tensor_load(self, inputs, batchsize):

        input_tensor = torch.stack([item[0] for item in inputs])
        label_tensor = torch.tensor([(item[1]) for item in inputs])
        data_set = TensorDataset(input_tensor, label_tensor)
        data_loader = DataLoader(data_set, batch_size = batchsize, shuffle= True)
        return data_loader

    def predict(self, model_input: torch.Tensor):
        """ Predicts a label for an input.

        Inputs:
            model_input (tensor): Input data for an example or a batch of examples.

        Returns:
            The predicted class.    

        """
        # TODO: Implement prediction for an input.

        self.eval()
        model_input = model_input.long()
        # print(model_input.shape)
            
        with torch.no_grad(): 
            model_input = model_input.unsqueeze(0) 
            outputs = self.forward(model_input)
            _, predicted_classes = torch.max(outputs, dim=1)
            # predicted_classes = predicted_classes.item()  
            # print(predicted_classes)
        return predicted_classes

    def calculate_accuracy(model, data_loader):
        model.eval()  
        correct_predictions = 0
        total_predictions = 0

        with torch.no_grad(): 
            for inputs, labels in data_loader:
                outputs = model(inputs)  
                _, predicted = torch.max(outputs, 1)  
                total_predictions += labels.size(0)  
                correct_predictions += (predicted == labels).sum().item()  

        accuracy = correct_predictions / total_predictions  
        return accuracy


    def learn(self, training_data, val_data, loss_fct, optimizer, num_epochs, lr) -> None:
        """ Trains the MLP.

        Inputs:
            training_data: (input->tensor, label)
            val_data: Validation data.
            loss_fct: The loss function.
            optimizer: The optimization method.
            num_epochs: The number of training epochs.
        """
        # TODO: Implement the training of this model.
        train_data_loader = self.tensor_load(training_data, batchsize = 1024)
        val_data_loader = self.tensor_load(val_data, batchsize = 1024)
       
        plot_loss = []
        plot_accuracy = []

        for epoch in range(num_epochs):
            self.train()
            train_loss = 0.0
            correct_train = 0
            total_train = 0

            for batch_inputs, batch_labels in train_data_loader:
                optimizer.zero_grad()
                outputs = self.forward(batch_inputs)
                # print(outputs)
                # print(batch_labels)
                loss = loss_fct(outputs, batch_labels)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

                _, predicted = torch.max(outputs.data, 1)
                total_train += batch_labels.size(0)
                correct_train += (predicted == batch_labels).sum().item()
            
            plot_loss.append(train_loss)

            train_accuracy = 100 * correct_train / total_train

            

            self.eval()
            val_loss = 0.0
            correct_val = 0
            total_val = 0

            with torch.no_grad():
                for batch_inputs, batch_labels in val_data_loader:
                    outputs = self.forward(batch_inputs)
                    loss = loss_fct(outputs, batch_labels)
                    val_loss += loss.item()

                    _, predicted = torch.max(outputs, 1)
                    total_val += batch_labels.size(0)
                    correct_val += (predicted == batch_labels).sum().item()

            val_accuracy = 100 * correct_val / total_val

            plot_accuracy.append(val_accuracy / 100)

            print(f"Epoch {epoch}, Training Loss: {train_loss:.4f}, Training Accuracy: {train_accuracy:.2f}%, "
                f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%.")
        
        # plt.figure(figsize=(20,10))
        # plt.title(f"MLP {data_type} Training Loss & Validation Accuracy")

        # plt.subplot(1, 2, 1)  
        # plt.plot(plot_loss, label='Train Loss', color='red')
        # plt.xlabel('Epochs')
        # plt.ylabel('Loss')
        # plt.title('Training Loss')
        # plt.legend()

        # plt.subplot(1, 2, 2)  
        # plt.plot(plot_accuracy, label='Validation Accuracy', color='blue')
        # plt.xlabel('Epochs')
        # plt.ylabel('Accuracy')
        # plt.title('Validation Accuracy')
        # plt.legend()

        # plt.tight_layout() 
        # plt.show()




    def forward(self, model_input):
        model_input = model_input.long() 
        model_input = self.embedding(model_input) 
        # print(model_input.shape)
        model_input = torch.mean(model_input, dim=1)
        model_input = self.mlp_1(model_input)
        return model_input 


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='MultiLayerPerceptron model')
    parser.add_argument('-d', '--data', type=str, default='sst2',
                        help='Dataset')
    parser.add_argument('-f', '--features', type=str, default='feature_name', help='Feature type')
    parser.add_argument('-m', '--model', type=str, default='mlp', help='Model type')
    args = parser.parse_args()

    data_type = args.data
    feature_type = args.features
    model_type = args.model

    train_data, val_data, dev_data, test_data = load_data(data_type, feature_type, model_type)
    
    all_inputs = [input for input, _ in train_data] + [input for input, _ in val_data]
    max_index = torch.cat(all_inputs).max().item() 
    vocab_size = int(max_index) + 1

    
    global input_length_init
    input_length_init = len(train_data[0][0])

    # print(input_length)
    
    labels = [int(label) for _, label in train_data]
    num_of_classes = len(set(labels))
    
    print(vocab_size)
    print(num_of_classes)

    # Train the model using the training data.
    model = MultilayerPerceptronModel(num_of_classes, vocab_size,  embedding_dim= 128)

    # Parameters
    loss_fct = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=0.01)


    num_epochs = 100

    lr = 0.01

    print("Training the model...")
    # Note: ensure you have all the inputs to the arguments.

    model.learn(train_data, val_data, loss_fct, optimizer, num_epochs, lr)

    # Predict on the development set. 
    # Note: if you used a dataloader for the dev set, you need to adapt the code accordingly.
    dev_accuracy = evaluate(model,
                            dev_data,
                            os.path.join("results", f"mlp_{data_type}_{feature_type}_dev_predictions.csv"))
    
    print(f"The accuracy in development set is {dev_accuracy}")

    # Predict on the test set.
    # Note: We don't provide labels for test, so the returned value from this
    # call shouldn't make sense.
    # evaluate(model,
    #          test_data,
    #          os.path.join("results", f"mlp_{data_type}_{feature_type}_test_predictions.csv"))
    
    # evaluate(model,
    #          test_data,
    #          os.path.join("results", f"mlp_{data_type}_test_predictions.csv"))
    
        
    # python multiLayer_perceptron.py -d 'newsgroups' -f 'feature_name' -m 'mlp'
    #python  multiLayer_perceptron.py -d 'newsgroups' -f 'feature_name' -m 'mlp'
    #python  multiLayer_perceptron.py -d 'newsgroups' -f 'feature_name' -m 'mlp'
    #python  multiLayer_perceptron.py -d 'newsgroups' -f 'bigram' -m 'mlp'
    #python  multiLayer_perceptron.py -d sst2 -f 'bigram' -m 'mlp'