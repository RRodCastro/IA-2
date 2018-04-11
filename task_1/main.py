from collections import Counter
from time import sleep
from random import shuffle
from os import remove
from analyzer import Analyzer

# Read File
nFile = open("test_corpus.txt", 'r')
lines = nFile.readlines()
nFile.close()
shuffle(lines)
spam = 'spam'
ham = 'ham'
corpus_partition = {'training': 0.80, 'cross_validation': 0.10, 'test': 0.10}


def count_types(lines):
    """
    Input: Array of lines (ham\t message)
    Output: Dict with number of ham/spam messages
    """
    words_dict = {ham: 0, spam: 0}
    for line in lines:
        message = line.split("\t")
        words_dict[message[0]] += 1
    return words_dict


def split_document(lines, corpus_partition):
    """
    Input: lines of document, dict of the partition
    Out: training lines [], cross_validation lines [], test lines [] (uniform distribution)
    """
    type_count = count_types(lines)
    training, cross_validation, test = [], [], []
    types = {ham: 0, spam: 0}
    training_count, cross_validation_count, test = [], [], []
    training_count.append(types.copy())
    cross_validation_count.append(types.copy())
    for partition in corpus_partition:
        percent = corpus_partition[partition]
        corpus_partition[partition] = {
            ham: int(type_count[ham] * percent), spam: int(type_count[spam] * percent)}
    for line in lines:
        type = line.split("\t")[0]
        if (corpus_partition['training'][type] > training_count[0][type]):
            training_count[0][type] += 1
            training.append(line)
        elif(corpus_partition['cross_validation'][type] > cross_validation_count[0][type]):
            cross_validation_count[0][type] += 1
            cross_validation.append(line)
        else:
            test.append(line)
    return training, cross_validation, test


def get_success_of_data(k, threshold, analyzer, data):
    """
    Input: k-factor, maximum difference in prob, analyzer object, a data to analyse
    Output: Percent of success messages that where tagged correctly
    """
    success = 0
    total = 0
    for line in data:
        message = line.split('\t')
        temp = float(analyzer.get_probabilty_from_message(message[1], k))
        if ((temp > threshold and message[0] == spam)
                or (temp <= threshold and message[0] == ham)):
            success += 1
        total += 1
    return success/total


def maximize_k(threshold, analyzer, data, max_iterations):
    probs = []
    for i in range(1, max_iterations):
        probs.append(get_success_of_data(i, 0.5, analyzer, data))
    return probs.index(max(probs))+1


def evaluate(k, threshold, analyzer, filename):
    """
    Input: k-factor,maximum difference in prob, analyzer object, a file
    Output: .txt with classfied messages
    """
    nFile = open(filename+".txt", 'r')
    lines = nFile.readlines()
    nFile.close()
    try:
        remove('result')
    except OSError:
        pass
    f = open('result', 'w')
    for line in lines:
        # print(line)
        #message = line.split("\t")
        temp = float(analyzer.get_probabilty_from_message(
            line, k))
        if (temp > threshold):
            f.write('spam\t'+line)
        else:
            f.write('ham\t'+line)
    f.close()


training, cross_validation, test = split_document(lines, corpus_partition)
analyzer = Analyzer(training)
best_k = 1
#best_k = (maximize_k(0.5, analyzer, cross_validation, 100))
#print(get_success_of_data(best_k, 0.5, analyzer, cross_validation))
#print(get_success_of_data(best_k, 0.5, analyzer, test))
#filename = input("Nombre de archivo a evaluar:\n")
evaluate(best_k, 0.5, analyzer, "test_sms")
