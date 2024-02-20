from typing import List, Any, Tuple
import os
from newsgroups import newsgroups_data_loader
from sst2 import sst2_data_loader


# def save_results(predictions: List[Any], results_path: str) -> None:
#     """ Saves the predictions to a file.

#     Inputs:
#         predictions (list of predictions, e.g., string)
#         results_path (str): Filename to save predictions to
#     """
#     # TODO: Implement saving of the results.

#     with open(results_path, 'w') as file:

#         if results_path == os.path.join("results", f"perceptron_sst2_test_predictions.csv"):
#             file.write('id,label\n')
#             for idx, prediction in enumerate(predictions):
#                 file.write(f"{idx},{str(prediction)}\n")

#         file.write('id,label\n')
#         for i, prediction in enumerate(predictions):
#             file.write(f"{i},{str(prediction)}\n")

def save_results(predictions: List[Any], results_path: str) -> None:
    """ Saves the predictions to a file.

    Inputs:
        predictions (list of predictions, e.g., string)
        results_path (str): Filename to save predictions to
    """
    # TODO: Implement saving of the results.
    feature_type = "feature_name"

    with open(results_path, 'w') as file:

        if results_path == os.path.join("results", f"perceptron_sst2_test_predictions.csv"):
            file.write('id,label\n')
            for idx, prediction in enumerate(predictions):
                file.write(f"{idx},{str(prediction)}\n")
        
        if results_path == os.path.join("results", f"mlp_sst2_test_predictions.csv"):
            file.write('id,label\n')
            for idx, prediction in enumerate(predictions):
                file.write(f"{idx},{str(prediction.item())}\n")

        label_dict = {'soc.religion.christian': 0, 'sci.electronics': 1, 'comp.sys.mac.hardware': 2, 
                    'alt.atheism': 3, 'rec.autos': 4, 'comp.os.ms-windows.misc': 5, 'sci.med': 6, 
                    'talk.politics.mideast': 7, 'rec.sport.hockey': 8, 'sci.crypt': 9, 'talk.politics.guns': 10, 
                    'comp.sys.ibm.pc.hardware': 11, 'comp.windows.x': 12, 'talk.politics.misc': 13, 
                    'talk.religion.misc': 14, 'rec.motorcycles': 15, 'misc.forsale': 16, 
                    'rec.sport.baseball': 17, 'sci.space': 18, 'comp.graphics': 19}
            
        reverted_label_dict = {v: k for k, v in label_dict.items()}

        if results_path == os.path.join("results", f"perceptron_newsgroups_test_predictions.csv"):
            file.write('id,newsgroup\n')
            for idx, prediction in enumerate(predictions):
                file.write(f"{idx},{str(reverted_label_dict[prediction])}\n")
        
        if results_path == os.path.join("results", f"mlp_newsgroups_test_predictions.csv"):
            file.write('id,newsgroup\n')
            for idx, prediction in enumerate(predictions):
                file.write(f"{idx},{str(reverted_label_dict[prediction.item()])}\n")
            
        if results_path == os.path.join("results", f"mlp_sst2_{feature_type}_dev_predictions.csv"):
            file.write('id,newsgroup\n')
            for idx, prediction in enumerate(predictions):
                file.write(f"{idx},{str(reverted_label_dict[prediction.item()])}\n")


        


def compute_accuracy(labels: List[Any], predictions: List[Any]) -> float:
    """ Computes the accuracy given some predictions and labels.

    Inputs:
        labels (list): Labels for the examples.
        predictions (list): The predictions.
    Returns:
        float representing the % of predictions that were true.
    """
    if len(labels) != len(predictions):
        raise ValueError("Length of labels (" + str(len(labels)) + " not the same as " \
                         "length of predictions (" + str(len(predictions)) + ".")
    # TODO: Implement accuracy computation.

    hit = 0
    n = len(labels)

    for i in range(n):
        if labels[i] == predictions[i]:
            hit += 1
        # else: print(i, predictions[i])
    result = hit / n
    return result


def evaluate(model: Any, data: List[Tuple[Any, Any]], results_path: str) -> float:
    """ Evaluates a dataset given the model.

    Inputs:
        model: A model with a prediction function.
        data: Suggested type is (list of pair), where each item is a training
            examples represented as an (input, label) pair. And when using the
            test data, your label can be some null value.
        results_path (str): A filename where you will save the predictions.
    """

    predictions = [model.predict(example[0]) for example in data]
    save_results(predictions, results_path)

    return compute_accuracy([example[1] for example in data], predictions)


def load_data(data_type: str, feature_type: str, model_type: str):
    """ Loads the data.

    Inputs:
        data_type: The type of data to load.
        feature_type: The type of features to use.
        model_type: The type of model to use.
        
    Returns:
        Training, validation, development, and testing data, as well as which kind of data
            was used.
    """
    data_loader = None
    if data_type == "newsgroups":
        data_loader = newsgroups_data_loader
    elif data_type == "sst2":
        data_loader = sst2_data_loader
    
    assert data_loader, "Choose between newsgroups or sst2 data. " \
                        + "data_type was: " + str(data_type)

    # Load the data. 
    train_data, val_data, dev_data, test_data = data_loader("data/" + data_type + "/train/train_data.csv",
                                                            "data/" + data_type + "/train/train_labels.csv",
                                                            "data/" + data_type + "/dev/dev_data.csv",
                                                            "data/" + data_type + "/dev/dev_labels.csv",
                                                            "data/" + data_type + "/test/test_data.csv",
                                                            feature_type,
                                                            model_type)

    return train_data, val_data, dev_data, test_data
