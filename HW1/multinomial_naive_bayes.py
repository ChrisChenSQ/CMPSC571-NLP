import numpy as np
from linear_classifier import LinearClassifier


class MultinomialNaiveBayes(LinearClassifier):

    def __init__(self):
        LinearClassifier.__init__(self)
        self.trained = False
        self.likelihood = 0
        self.prior = 0
        self.smooth = True
        self.smooth_param = 1

    def train(self, x, y):
        # n_docs = no. of documents
        # n_words = no. of unique words    
        n_docs, n_words = x.shape

        # classes = a list of possible classes
        classes = np.unique(y)

        # n_classes = no. of classes
        n_classes = np.unique(y).shape[0]

        # initialization of the prior and likelihood variables
        prior = np.zeros(n_classes)
        likelihood = np.zeros((n_words, n_classes))

        # TODO: This is where you have to write your code!
        # You need to compute the values of the prior and likelihood parameters
        # and place them in the variables called "prior" and "likelihood".
        # Examples:
        # prior[0] is the prior probability of a document being of class 0
        # likelihood[4, 0] is the likelihood of the fifth(*) feature being
        # active, given that the document is of class 0
        # (*) recall that Python starts indices at 0, so an index of 4
        # corresponds to the fifth feature!

        ###########################

        # YOUR CODE HERE
        positive_class = x[np.where(y.reshape(1, -1).reshape(-1) == 0)]
        negative_class = x[np.where(y.reshape(1, -1).reshape(-1) == 1)]

        positve_sum = np.sum(positive_class, axis=0)
        negative_sum = np.sum(negative_class, axis=0)

        positve_total_sum = np.sum(positve_sum)
        negative_total_sum = np.sum(negative_sum)

        print(positve_sum.shape)
        print(negative_sum)

        for i in range(n_words):
            if positve_sum[i] / positve_total_sum > negative_sum[i] / negative_total_sum:
                likelihood[i] = [i, 0]
            else:
                likelihood[i] = [i, 1]

        prior[0] = positive_class.shape[0] / n_docs
        prior[1] = negative_class.shape[0] / n_docs

        print(prior)

        ###########################

        params = np.zeros((n_words + 1, n_classes))
        for i in range(n_classes):
            # log probabilities
            params[0, i] = np.log(prior[i])
            with np.errstate(divide='ignore'):  # ignore warnings
                params[1:, i] = np.nan_to_num(np.log(likelihood[:, i]))
        self.likelihood = likelihood
        self.prior = prior
        self.trained = True
        return params
