""" Perceptron model for Assignment 1: Starter code.

You can change this code while keeping the function headers.
"""
import os
import sys
import argparse
from typing import Dict, List

from util import evaluate, load_data



class PerceptronModel():
    """ Perceptron model for classification.
    """
    def __init__(self, num_features: int, num_classes: int):
        """ Initializes the model.

        Inputs:
            num_features (int): The number of features.
            num_classes (int): The number of classes.
        """
        # TODO: Implement initialization of this model.
        self.weights: Dict[int, Dict[int, float]] = {class_index: {feature_index: 0.0 for feature_index in range(num_features)} for class_index in range(num_classes)}
        self.num_classes = num_classes
        self.num_features = num_features
    
    def score(self, model_input: Dict, class_id: int):
        """ Compute the score of a class given the input.
        Inputs:
            model_input (features): Input data for an example
            class_id (int): Class id.
        
        Returns:
            The output score.
        """
        # TODO: Implement scoring function.
        score = 0.0
        for feature_index, feature_value in model_input.items():
            score += self.weights[class_id][feature_index] * feature_value
        
        return score
        # pass

    def predict(self, model_input: Dict) -> int:
        """ Predicts a label for an input.

        Inputs:
            model_input (features): Input data for an example

        Returns:
            The predicted class.    
        """
        # TODO: Implement prediction for an input.
        pred_class = {} 
        for class_id in range(self.num_classes):
            pred_class[class_id] = self.score(model_input, class_id)
        return max(pred_class, key=pred_class.get)
    
    def update_parameters(self, model_input: Dict, prediction: int, target: int, lr: float) -> None:
        """ Update the model weights of the model using the perceptron update rule.

        Inputs:
            model_input (features): Input data for an example
            prediction: The predicted label.
            target: The true label.
            lr: Learning rate.
        """
        # TODO: Implement the parameter updates.
        if prediction != target:
            for feature_id, feature_value in model_input.items():
                self.weights[prediction][feature_id] -= lr * feature_value
                self.weights[target][feature_id] += lr * feature_value   
         
    def learn(self, training_data, val_data, num_epochs, lr) -> None:
        """ Perceptron model training.

        Inputs:
            training_data: Suggested type is (list of tuple), where each item can be
                a training example represented as an (input, label) pair or (input, id, label) tuple.
            val_data: Validation data.
            num_epochs: Number of training epochs.
            lr: Learning rate.
        """
        # TODO: Implement the training of this model.

        total = len(training_data)
        val_total = len(val_data)

        # print(training_data[0])

        for epoch in range(num_epochs):
            num_train = 0
            num_val = 0

            training_accuracy = 0
            val_accuracy = 0
        
            
            for input, label in training_data:
                prediction = self.predict(input)
                if prediction == label:
                    num_train += 1
                self.update_parameters(input, prediction, label, lr)
                training_accuracy = num_train / total
            
            for val_input, val_label in val_data:
                val_prediction = self.predict(val_input)
                if val_prediction == val_label:
                    num_val += 1
                val_accuracy = num_val / val_total  
            

            
            print(f"Epoch {epoch}, Training Accuracy:{training_accuracy}, Validation Accuracy:{val_accuracy}")
        
        print("Training Finsihed!")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Perceptron model')
    parser.add_argument('-d', '--data', type=str, default='sst2',
                        help='Dataset')
    parser.add_argument('-f', '--features', type=str, default='feature_name', help='Feature type')
    parser.add_argument('-m', '--model', type=str, default='perceptron', help='Model type')
    args = parser.parse_args()

    data_type = args.data
    feature_type = args.features
    model_type = args.model
    print(feature_type)
    train_data, val_data, dev_data, test_data = load_data(data_type, feature_type, model_type)
    # train_data: dict-> ([id] : 0 / 1 ), lable
    # print(train_data)

    if train_data:
        num_of_features = len(train_data[0][0])
    else:
        num_of_features = 0

    labels = [label for _, label in train_data]
    num_of_classes = len(set(labels))

    # print(num_of_classes, num_of_features)
    if data_type == "sst2":
        num_epochs = 5
        lr = 3
    else:
        num_epochs = 5
        lr = 3
    # Train the model using the training data.
    model = PerceptronModel(num_of_features, num_of_classes)

    print("Training the model...")
    # Note: ensure you have all the inputs to the arguments.
    model.learn(train_data, val_data, num_epochs, lr)

    # Predict on the development set. 
    dev_accuracy = evaluate(model,
                            dev_data,
                            os.path.join("results", f"perceptron_{data_type}_{feature_type}_dev_predictions.csv"))
    
    print(f"development accuracuy is {dev_accuracy}")

    # Predict on the test set.
    # Note: We don't provide labels for test, so the returned value from this
    # call shouldn't make sense.
    # evaluate(model,
    #          test_data,
    #          os.path.join("results", f"perceptron_{data_type}_{feature_type}_test_predictions.csv"))
    
    # evaluate(model,
    #         test_data,
    #         os.path.join("results", f"perceptron_{data_type}_test_predictions.csv"))


    #python perceptron.py -d newsgroups -f feature_name -m perceptron
    #python perceptron.py -d newsgroups -f bigram -m perceptron
    #python perceptron.py -d sst2 -f bigram -m perceptron
    # python perceptron.py -d sst2 -f whole -m perceptron
    # python perceptron.py -d sst2 -f idf_unigram -m perceptron
    # python perceptron.py -d sst2 -f idf_bigram -m perceptron
    #python perceptron.py -d sst2 -f feature_name -m perceptron
    #python perceptron.py -d sst2 -f 'whole' -m perceptron
    # python perceptron.py -d newsgroups -f 'whole' -m perceptron
    # python perceptron.py -d newsgroups -f 'idf_unigram' -m perceptron
    # python perceptron.py -d newsgroups -f 'bigram' -m perceptron
