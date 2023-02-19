import codecs
import numpy as np

class SentimentCorpus:
    
    def __init__(self, train_per=0.8, dev_per=0, test_per=0.2):
        '''
        prepare dataset
        1) build feature dictionaries
        2) split data into train/dev/test sets 
        '''
        train_X, train_y, test_X, test_y, feat_dict, feat_counts = build_train_dicts()
        self.train_nr_instances = train_y.shape[0]
        self.train_nr_features = train_X.shape[1]
        self.train_X = train_X
        self.train_y = train_y
        self.test_X = test_X
        self.test_y = test_y
        self.feat_dict = feat_dict
        self.feat_counts = feat_counts



def build_train_dicts():
    '''
    builds feature dictionaries
    ''' 
    feat_counts = {}

    # build feature dictionary with counts
    train_nr_pos = 0
    with codecs.open("positive.train.review", 'r', 'utf8') as pos_file:
        for line in pos_file:
            train_nr_pos += 1
            toks = line.split(" ")
            for feat in toks[0:-1]:
                name, counts = feat.split(":")
                if name not in feat_counts:
                    feat_counts[name] = 0
                feat_counts[name] += int(counts)
    
    train_nr_neg = 0
    with codecs.open("negative.train.review", 'r', 'utf8') as neg_file:
        for line in neg_file:
            train_nr_neg += 1
            toks = line.split(" ")
            for feat in toks[0:-1]:
                name, counts = feat.split(":")
                if name not in feat_counts:
                    feat_counts[name] = 0
                feat_counts[name] += int(counts)

    print("Total vocabulary: {}".format(len(feat_counts)))

    test_nr_pos = 0
    with codecs.open("positive.test.review", 'r', 'utf8') as pos_file:
        for line in pos_file:
            test_nr_pos += 1
            toks = line.split(" ")
            for feat in toks[0:-1]:
                name, counts = feat.split(":")
                if name not in feat_counts:
                    feat_counts[name] = 0
                feat_counts[name] += int(counts)

    test_nr_neg = 0
    with codecs.open("negative.test.review", 'r', 'utf8') as neg_file:
        for line in neg_file:
            test_nr_neg += 1
            toks = line.split(" ")
            for feat in toks[0:-1]:
                name, counts = feat.split(":")
                if name not in feat_counts:
                    feat_counts[name] = 0
                feat_counts[name] += int(counts)

    # remove all features that occur less than 5 (threshold) times
    to_remove = []
    for key, value in feat_counts.items():
        if value < 5:
            to_remove.append(key)
    for key in to_remove:
        del feat_counts[key]

    # map feature to index
    feat_dict = {}
    i = 0
    for key in feat_counts.keys():
        feat_dict[key] = i
        i += 1

    nr_feat = len(feat_counts)
    train_X = np.zeros((train_nr_pos + train_nr_neg, nr_feat), dtype=float)
    train_y = np.vstack((np.zeros([train_nr_pos,1], dtype=int), np.ones([train_nr_neg,1], dtype=int)))
    test_X = np.zeros((test_nr_neg + test_nr_pos, nr_feat), dtype=float)
    test_y = np.vstack((np.zeros([test_nr_pos, 1], dtype=int), np.ones([test_nr_neg, 1], dtype=int)))
    
    with codecs.open("positive.train.review", 'r', 'utf8') as pos_file:
        nr_pos = 0
        for line in pos_file:
            toks = line.split(" ")
            for feat in toks[0:-1]:
                name, counts = feat.split(":")
                if name in feat_dict:
                    train_X[nr_pos,feat_dict[name]] = int(counts)
            nr_pos += 1
        
    with codecs.open("negative.train.review", 'r', 'utf8') as neg_file:
        nr_neg = 0
        for line in neg_file:
            toks = line.split(" ")
            for feat in toks[0:-1]:
                name, counts = feat.split(":")
                if name in feat_dict:
                    train_X[nr_pos+nr_neg,feat_dict[name]] = int(counts)
            nr_neg += 1

    with codecs.open("positive.test.review", 'r', 'utf8') as pos_file:
        nr_pos = 0
        for line in pos_file:
            toks = line.split(" ")
            for feat in toks[0:-1]:
                name, counts = feat.split(":")
                if name in feat_dict:
                    test_X[nr_pos, feat_dict[name]] = int(counts)
            nr_pos += 1

    with codecs.open("negative.test.review", 'r', 'utf8') as neg_file:
        nr_neg = 0
        for line in neg_file:
            toks = line.split(" ")
            for feat in toks[0:-1]:
                name, counts = feat.split(":")
                if name in feat_dict:
                    test_X[nr_pos + nr_neg, feat_dict[name]] = int(counts)
            nr_neg += 1

    # shuffle the order, mix positive and negative examples
    new_order = np.arange(train_nr_pos + train_nr_neg)
    np.random.seed(0) # set seed
    np.random.shuffle(new_order)
    train_X = train_X[new_order,:]
    train_y = train_y[new_order,:]

    return train_X, train_y, test_X, test_y, feat_dict, feat_counts
