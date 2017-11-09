import csv
import sys 

def evaluate_classifier(classifier, eval_set, batch_size):
    """
    Function to get accuracy and cost of the model, evaluated on a chosen dataset.

    classifier: the model's classfier, it should return genres, logit values, and cost for a given minibatch of the evaluation dataset
    eval_set: the chosen evaluation set, for eg. the dev-set
    batch_size: the size of minibatches.
    """
    correct = 0
    genres, hypotheses, cost = classifier(eval_set)
    cost = cost / batch_size
    full_batch = int(len(eval_set) / batch_size) * batch_size

    for i in range(full_batch):
        hypothesis = hypotheses[i]
        if hypothesis == eval_set[i]['label']:
            correct += 1        
    return correct / float(len(eval_set)), cost

def evaluate_classifier_genre(classifier, eval_set, batch_size):
    """
    Function to get accuracy and cost of the model by genre, evaluated on a chosen dataset. It returns a dictionary of accuracies by genre and cost for the full evaluation dataset.
    
    classifier: the model's classfier, it should return genres, logit values, and cost for a given minibatch of the evaluation dataset
    eval_set: the chosen evaluation set, for eg. the dev-set
    batch_size: the size of minibatches.
    """
    genres, hypotheses, cost = classifier(eval_set)
    correct = dict((genre,0) for genre in set(genres))
    count = dict((genre,0) for genre in set(genres))
    cost = cost / batch_size
    full_batch = int(len(eval_set) / batch_size) * batch_size

    for i in range(full_batch):
        hypothesis = hypotheses[i]
        genre = genres[i]
        if hypothesis == eval_set[i]['label']:
            correct[genre] += 1.
        count[genre] += 1.

        if genre != eval_set[i]['genre']:
            print 'welp!'

    accuracy = {k: correct[k]/count[k] for k in correct}

    return accuracy, cost

def evaluate_final(restore, classifier, eval_sets, batch_size):
    """
    Function to get percentage accuracy of the model, evaluated on a set of chosen datasets.
    
    restore: a function to restore a stored checkpoint
    classifier: the model's classfier, it should return genres, logit values, and cost for a given minibatch of the evaluation dataset
    eval_set: the chosen evaluation set, for eg. the dev-set
    batch_size: the size of minibatches.
    """
    restore(best=True)
    percentages = []
    for eval_set in eval_sets:
        genres, hypotheses, cost = classifier(eval_set)
        correct = 0
        cost = cost / batch_size
        full_batch = int(len(eval_set) / batch_size) * batch_size

        for i in range(full_batch):
            hypothesis = hypotheses[i]
            if hypothesis == eval_set[i]['label']:
                correct += 1      
        percentages.append(correct / float(len(eval_set)))  
    return percentages


def predictions_kaggle(classifier, eval_set, batch_size, name):
    """
    Get comma-separated CSV of predictions.
    Output file has two columns: pairID, prediction
    """
    INVERSE_MAP = {
    0: "entailment",
    1: "neutral",
    2: "contradiction"
    }

    hypotheses = classifier(eval_set)
    predictions = []

    correct_result_count = 0
    total_count = len(eval_set)
    for i in range(len(eval_set)):
        hypothesis = hypotheses[i]
        prediction = INVERSE_MAP[hypothesis]
        pairID = eval_set[i]["pairID"]
        sentence1 = eval_set[i]["sentence1"]
        sentence2 = eval_set[i]["sentence2"]
        actual_label = eval_set[i]["gold_label"]
        if actual_label == prediction:
            correct_result_count += 1
        predictions.append((pairID, prediction, actual_label, sentence1, sentence2))

    #predictions = sorted(predictions, key=lambda x: int(x[0]))

    f = open( name + '_predictions.csv', 'wb')
    w = csv.writer(f, delimiter = ',')
    w.writerow(['pairID','gold_label'])
    for example in predictions:
        w.writerow(example)
    w.writerow("correct results = " + str(correct_result_count)+", Total count = "+str(total_count))
    w.writerow("accuracy = " + str(float(correct_result_count)/float(total_count)))
    f.close()
