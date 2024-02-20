import re
import random
import csv
import heapq
import torch
import sys
from collections import Counter
import math

csv.field_size_limit(sys.maxsize)

def newsgroups_featurize(train_data, val_data, dev_data, test_data, feature_type):
    """ Featurizes an input for the newsgroups domain.

    Inputs:
        train_data: The training data.
        val_data: The validation data.
        dev_data: The development data.
        test_data: The test data.
        feature_type: Type of feature to be used.
    """
    # TODO: Implement featurization of input.
    stop_set = set()

    with open("stopwords.txt", "r", encoding= "utf-8") as file:
        for line in file:
            stop_set.add(line.strip())

    train_clean = [[word for word in clean_text(text).split() if word not in stop_set] for text in train_data]
    val_clean = [[word for word in clean_text(text).split() if word not in stop_set] for text in val_data]
    dev_clean = [[word for word in clean_text(text).split() if word not in stop_set] for text in dev_data]
    test_clean = [[word for word in clean_text(text).split() if word not in stop_set] for text in test_data]

    bow = train_clean + val_clean

    n_gram = 1

    
    if feature_type == 'idf_bigram' or feature_type == 'idf_unigram' or feature_type == 'whole':
        size = 1000
    else:
        size = 10000
    
    bow_dict = bag_of_words(bow, n_gram, size)
    train_vector = []
    val_vector = []
    dev_vector = []
    test_vector = []

    if feature_type == "feature_name":
        for text in train_clean:
            train_vector.append(vectorize(text, bow_dict, n_gram))
        
        for text in val_clean:
            val_vector.append(vectorize(text, bow_dict, n_gram))
        
        for text in dev_clean:
            dev_vector.append(vectorize(text, bow_dict, n_gram))
        
        for text in test_clean:
            test_vector.append(vectorize(text, bow_dict, n_gram))

        print(f"Number of Features:{len(train_vector[0])}")

        return train_vector, val_vector, dev_vector, test_vector
    
    if feature_type == "bigram":
        n_gram = 2
        size = 100
        bi_dict = bag_of_words(bow, n_gram, size)

        for text in train_clean:
            train_vector.append(vectorize(text, bi_dict, n_gram) + vectorize(text, bow_dict, 1))
        
        for text in val_clean:
            val_vector.append(vectorize(text, bi_dict, n_gram) + vectorize(text, bow_dict, 1))
        
        for text in dev_clean:
            dev_vector.append(vectorize(text, bi_dict, n_gram) + vectorize(text, bow_dict, 1))
        
        for text in test_clean:
            test_vector.append(vectorize(text, bi_dict, n_gram) + vectorize(text, bow_dict, 1))
        
        print(f"Number of Features:{len(train_vector[0])}")
        return train_vector, val_vector, dev_vector, test_vector
    
    if feature_type == "idf_bigram":

        word_set = bow_dict.keys()
        idf_dict = compute_idf(bow, word_set)
        
        n_gram = 2
        size = 100
        bi_dict = bag_of_words(bow, n_gram, size)

        for text in train_clean:
            train_vector.append(vectorize(text, bi_dict, n_gram) +  compute_tf_idf_for_doc(text, idf_dict, word_set))
        
        for text in val_clean:
            val_vector.append(vectorize(text, bi_dict, n_gram) + compute_tf_idf_for_doc(text, idf_dict, word_set))
        
        for text in dev_clean:
            dev_vector.append(vectorize(text, bi_dict, n_gram) + compute_tf_idf_for_doc(text, idf_dict, word_set))
        
        for text in test_clean:
            test_vector.append(vectorize(text, bi_dict, n_gram) +  compute_tf_idf_for_doc(text, idf_dict, word_set))
        
        print(f"Number of Features:{len(train_vector[0])}")
        return train_vector, val_vector, dev_vector, test_vector
    
    if feature_type == "idf_unigram":

        word_set = bow_dict.keys()
        idf_dict = compute_idf(bow, word_set)
        
        for text in train_clean:
            train_vector.append(vectorize(text, bow_dict, 1) + compute_tf_idf_for_doc(text, idf_dict, word_set))
        
        for text in val_clean:
            val_vector.append(vectorize(text, bow_dict, 1) + compute_tf_idf_for_doc(text, idf_dict, word_set))
        
        for text in dev_clean:
            dev_vector.append(vectorize(text, bow_dict, 1) + compute_tf_idf_for_doc(text, idf_dict, word_set))
        
        for text in test_clean:
            test_vector.append(vectorize(text, bow_dict, 1) + compute_tf_idf_for_doc(text, idf_dict, word_set))
        
        print(f"Number of Features:{len(train_vector[0])}")
        return train_vector, val_vector, dev_vector, test_vector


    if feature_type == "whole":

        word_set = bow_dict.keys()
        idf_dict = compute_idf(bow, word_set)

        n_gram = 2
        size = 100
        bi_dict = bag_of_words(bow, n_gram, size)

        for text in train_clean:
            train_vector.append(vectorize(text, bi_dict, n_gram) + vectorize(text, bow_dict, 1) + compute_tf_idf_for_doc(text, idf_dict, word_set))
        
        for text in val_clean:
            val_vector.append(vectorize(text, bi_dict, n_gram) + vectorize(text, bow_dict, 1)+ compute_tf_idf_for_doc(text, idf_dict, word_set))
        
        for text in dev_clean:
            dev_vector.append(vectorize(text, bi_dict, n_gram) + vectorize(text, bow_dict, 1)+ compute_tf_idf_for_doc(text, idf_dict, word_set))
        
        for text in test_clean:
            test_vector.append(vectorize(text, bi_dict, n_gram) + vectorize(text, bow_dict, 1)+ compute_tf_idf_for_doc(text, idf_dict, word_set))
            
        print(f"Number of Features:{len(train_vector[0])}")
        return train_vector, val_vector, dev_vector, test_vector
    

    # if feature_type == "feature_name":
    #     for text in train_clean:
    #         train_vector.append(vectorize(text, bow_dict, n_gram))
        
    #     for text in val_clean:
    #         val_vector.append(vectorize(text, bow_dict, n_gram))
        
    #     for text in dev_clean:
    #         dev_vector.append(vectorize(text, bow_dict, n_gram))
        
    #     for text in test_clean:
    #         test_vector.append(vectorize(text, bow_dict, n_gram))

    #     print(f"Number of Features:{len(train_vector[0])}")
    
    # if feature_type == "bigram":
    #     n_gram = 2
    #     size = 100
    #     bi_dict = bag_of_words(bow, n_gram, size)

    #     for text in train_clean:
    #         train_vector.append(vectorize(text, bi_dict, n_gram) + vectorize(text, bow_dict, 1))
        
    #     for text in val_clean:
    #         val_vector.append(vectorize(text, bi_dict, n_gram) + vectorize(text, bow_dict, 1))
        
    #     for text in dev_clean:
    #         dev_vector.append(vectorize(text, bi_dict, n_gram) + vectorize(text, bow_dict, 1))
        
    #     for text in test_clean:
    #         test_vector.append(vectorize(text, bi_dict, n_gram) + vectorize(text, bow_dict, 1))
        
    #     print(f"Number of Features:{len(train_vector[0])}")
    # return train_vector, val_vector, dev_vector, test_vector

 

def newsgroups_data_loader(train_data_filename: str,
                           train_labels_filename: str,
                           dev_data_filename: str,
                           dev_labels_filename: str,
                           test_data_filename: str,
                           feature_type: str,
                           model_type: str):
    """ Loads the data.

    Inputs:
        train_data_filename: The filename of the training data.
        train_labels_filename: The filename of the training labels.
        dev_data_filename: The filename of the development data.
        dev_labels_filename: The filename of the development labels.
        test_data_filename: The filename of the test data.
        feature_type: The type of features to use.
        model_type: The type of model to use.

    Returns:
        Training, validation, dev, and test data, all represented as a list of (input, data_id, label) format.

        Suggested: for test data, put in some dummy value as the label.
    """
    # TODO: Load the data from the text format.

    def read_csv(file_text, file_label):

        data_text = {}
        data_label = {}

        with open(file_text, mode='r', encoding='utf-8') as file:
            csv_reader = csv.reader(file)
            next(csv_reader, None)
            for row in csv_reader:
                data_text[row[0]] = row[1]
        
        with open(file_label, mode='r', encoding='utf-8') as file:
            csv_reader = csv.reader(file)
            next(csv_reader, None)
            for row in csv_reader:
                data_label[row[0]] = row[1]
        
        data_merge = []
        merge_label = []

        for id, text in data_text.items():
            label = data_label.get(id)
            if label:
                data_merge.append(text)
                merge_label.append(label)

        return data_merge, merge_label
    
    train_data, train_label = read_csv(train_data_filename, train_labels_filename)
    dev_data, dev_label = read_csv(dev_data_filename, dev_labels_filename)

    label_dict = {'soc.religion.christian': 0, 'sci.electronics': 1, 'comp.sys.mac.hardware': 2, 
                  'alt.atheism': 3, 'rec.autos': 4, 'comp.os.ms-windows.misc': 5, 'sci.med': 6, 
                  'talk.politics.mideast': 7, 'rec.sport.hockey': 8, 'sci.crypt': 9, 'talk.politics.guns': 10, 
                  'comp.sys.ibm.pc.hardware': 11, 'comp.windows.x': 12, 'talk.politics.misc': 13, 
                  'talk.religion.misc': 14, 'rec.motorcycles': 15, 'misc.forsale': 16, 
                  'rec.sport.baseball': 17, 'sci.space': 18, 'comp.graphics': 19}

    trans_train = tran_labels(train_label, label_dict)
    trans_dev = tran_labels(dev_label, label_dict)

    # print(trans_train)

    train_label = trans_train
    dev_label = trans_dev

    paired = list(zip(train_data, train_label))
    random.shuffle(paired)

    val_size = int(len(paired) * 0.2)

    val_all = paired[:val_size]
    train_all = paired[val_size:]

    train_data, train_label = zip(*train_all)
    val_data, val_label = zip(*val_all)

    test_data = []

    with open(test_data_filename, mode='r', encoding='utf-8') as file:
        csv_reader = csv.reader(file)
        next(csv_reader, None)
        for row in csv_reader:
            test_data.append(row[1])
    


    if model_type == "perceptron":
    # TODO: Featurize the input data for all three splits.  
        train_vec, val_vec, dev_vec, test_vec = newsgroups_featurize(train_data, val_data, dev_data, test_data, feature_type)
        train_list = make_list(train_vec, train_label)
        val_list = make_list(val_vec, val_label)
        dev_list = make_list(dev_vec, dev_label)
        test_list = make_list(test_vec, None)
        return train_list, val_list, dev_list, test_list
    
    if model_type == "mlp":
        train_token, val_token, dev_token, test_token = embedding(train_data, val_data, dev_data, test_data, feature_type)

        train_dic = make_tensor(train_token, train_label)
        val_dic = make_tensor(val_token, val_label)
        dev_dic = make_tensor(dev_token, dev_label)
        test_dic = make_tensor(test_token, None) 
        return train_dic, val_dic, dev_dic, test_dic



def embedding(train_data, val_data, dev_data, test_data, feature_type):
    
    stop_set = set()

    with open("stopwords.txt", "r", encoding= "utf-8") as file:
        for line in file:
            stop_set.add(line.strip())
    
    train_clean = [[word for word in clean_text(text).split() if word not in stop_set] for text in train_data]
    val_clean = [[word for word in clean_text(text).split() if word not in stop_set] for text in val_data]
    dev_clean = [[word for word in clean_text(text).split() if word not in stop_set] for text in dev_data]
    test_clean = [[word for word in clean_text(text).split() if word not in stop_set] for text in test_data]

    # print(train_clean[0])

    ebd_list = train_clean + val_clean

    n_gram = 1
    input_length, ebd_dict = create_ebd(ebd_list)
    # print(ebd_dict)
    # print(input_length)

    train_token = look_up(train_clean, ebd_dict,  n_gram)
    val_token = look_up(val_clean, ebd_dict,  n_gram)
    dev_token = look_up(dev_clean, ebd_dict,  n_gram)
    test_token = look_up(test_clean, ebd_dict, n_gram)

    return train_token, val_token, dev_token, test_token
    
    # if feature_type == "bigram":
    #     n_gram = 1
    #     input_length, ebd_dict = create_ebd(ebd_list, 0, n_gram)
    #     train_token = look_up(train_clean, ebd_dict, input_length, n_gram)
    #     val_token = look_up(val_clean, ebd_dict, input_length, n_gram)
    #     dev_token = look_up(dev_clean, ebd_dict, input_length, n_gram)
    #     test_token = look_up(test_clean, ebd_dict, input_length, n_gram)

    #     n_gram = 2
    #     input_length, bi_dict = create_ebd(ebd_list, input_length, n_gram)

    #     bi_train_token = look_up(train_clean, bi_dict, input_length, n_gram)
    #     bi_val_token = look_up(val_clean, bi_dict, input_length, n_gram)
    #     bi_dev_token = look_up(dev_clean, bi_dict, input_length, n_gram)
    #     bi_test_token = look_up(test_clean, bi_dict, input_length, n_gram)

    #     concat_train_token = [x + y for x, y in zip(train_token, bi_train_token)]
    #     concat_val_token = [x + y for x, y in zip(val_token, bi_val_token)]
    #     concat_dev_token = [x + y for x, y in zip(dev_token, bi_dev_token)]
    #     concat_test_token = [x + y for x, y in zip(test_token, bi_test_token)]

    #     return concat_train_token, concat_val_token, concat_dev_token, concat_test_token

def clean_text(input):
    text_clean = re.sub(r'[^\w\s]|\d', ' ', input)
    text_clean = text_clean.lower()
    return text_clean

def bag_of_words(sentences, n_gram, size):

    bow_dict = {}

    for word_list in sentences:
        n = len(word_list)
        for i in range(n - n_gram + 1):
            gram_text = " ".join(word_list[i:i + n_gram])
            if gram_text not in bow_dict:
                bow_dict[gram_text] = 1
            else:
                bow_dict[gram_text] += 1
    
    bow = heapq.nlargest(size, bow_dict, key=bow_dict.get)
    values = [0] * size
    empty_dict = {key : value for key, value in zip(bow, values)}
    return empty_dict

def vectorize(input, dict, n_gram):
    n = len(input)
    vector = dict.copy()
    for i in range(n - n_gram + 1):
        text = " ".join(input[i:i + n_gram]) 
        
        if text in vector:
            vector[text] = 1
    return list(vector.values())

def make_list(input, label):
    if label is None:
        label = [0] * len(input)

    vect_list = []

    for i in range(len(input)):
        vect = input[i]
        dic = {}
        for j in range(len(vect)):
            dic[j] = vect[j]
        vect_list.append((dic, (label[i])))

    return vect_list

def tran_labels(input_list, label_dict):
    print(label_dict)
    result = []
    for input in input_list:
        result.append(label_dict[input])
    return result

def make_tensor(inputs, label):
    if label is None:
        label = [0] * len(inputs)
    
    tensor_list = []
    # print(label)
    for i in range(len(inputs)):
        vect = inputs[i]
        vect = torch.tensor(vect, dtype=torch.float32)
        tensor_list.append((vect,torch.tensor(label[i])))
    return tensor_list

def create_ebd(inputs):
    ebd_dict = {}
    ebd_dict["<pad>"] = 0
    ebd_dict["<unk>"] = 1
    i = 2
    max_len = 0
    for sentence in inputs:

        input_length = len(sentence)
        if input_length > max_len:
            max_len = input_length
            
        for word in sentence:
            if word not in ebd_dict:
                ebd_dict[word] = i
                i += 1
    
    return max_len, ebd_dict


def look_up(inputs, bd_dict, n_gram=1):
    max_len = 50
    ebd_list = []
    for sentence in inputs:
        line = [] 
        for word in sentence:
            if len(line) < max_len:
                if word in bd_dict:
                    line.append(bd_dict[word])
                else:
                    line.append(bd_dict["<unk>"])

        while len(line) < max_len:
            line.append(bd_dict["<pad>"])

        ebd_list.append(line)
    return ebd_list


def compute_idf(documents, vocab):
    idf = dict.fromkeys(vocab, 0)
    N = len(documents)
    
    for doc in documents:
        seen_words = set()
        for word in doc:
            if word in vocab and word not in seen_words:
                idf[word] += 1
                seen_words.add(word)
    
    for word, count in idf.items():
        idf[word] = math.log(N / float(count + 1))  
    return idf

def compute_tf(doc):
    tf = {}
    doc_count = Counter(doc)
    for word, count in doc_count.items():
        tf[word] = count / len(doc)
    return tf

def compute_tf_idf_for_doc(doc, idf_dict, vocab):
    tf_idf_vector = {}
    tf = compute_tf(doc)
    
    tf_idf_vector = [tf.get(word, 0) * idf_dict.get(word, 0) for word in vocab]
    
    return tf_idf_vector


# if __name__ == "__main__":

#     data_type = "newsgroups"
#     train_data,val_data, dev_data, test_data = newsgroups_data_loader("data/" + data_type + "/train/train_data.csv",
#                 "data/" + data_type + "/train/train_labels.csv",
#                 "data/" + data_type + "/dev/dev_data.csv",
#                 "data/" + data_type + "/dev/dev_labels.csv",
#                 "data/" + data_type + "/test/test_data.csv",
#                 None,
#                 "mlp")
    