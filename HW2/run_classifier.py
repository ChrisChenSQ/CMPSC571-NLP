import time

from sklearn.metrics import f1_score

from sentiment_reader import SentimentCorpus
from multinomial_naive_bayes import MultinomialNaiveBayes

if __name__ == '__main__':
    start = time.time()
    dataset = SentimentCorpus()
    nb = MultinomialNaiveBayes()
    
    params = nb.train(dataset.train_X, dataset.train_y)
    
    predict_train = nb.test(dataset.train_X, params)
    eval_train = nb.evaluate(predict_train, dataset.train_y)
    
    predict_test = nb.test(dataset.test_X, params)
    eval_test = nb.evaluate(predict_test, dataset.test_y)

    end = time.time()
    print("Total runtime is: {}".format(end-start))
    
    print("Accuracy on training set: %f, on test set: %f" % (eval_train, eval_test))
    print("The f1 score is: {}".format(f1_score(dataset.test_y, predict_test, average='macro')))


