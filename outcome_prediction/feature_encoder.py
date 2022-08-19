import pickle
import numpy as np
from sys import getsizeof
import time


class FeatureEncoder(object):

    def one_hot_encoding_xgb(self, df, coded_activity, coded_labels, maxlen):
        Outcomes_list = list(coded_labels.keys())
        feature_set = list(coded_activity.keys())
        feature_len = len(feature_set)
        dict_feature_char_to_int = dict((str(c), i) for i, c in enumerate(feature_set))
        outcome_char_to_int = {j: i for i, j in enumerate(Outcomes_list)}

        X_train = list()
        y_train = list()
        for i in range(0, len(df)):
            print("{}th among {}".format(i, len(df)))
            hist_len = len(str(df.at[i, "prefix"]).split("+"))
            merged_encoding = list()
            for j in range(hist_len):
                parsed_hist = str(df.at[i, "prefix"]).split("+")
                feature = parsed_hist[j]
                feature_int = dict_feature_char_to_int[feature]
                onehot_encoded_feature = [0 for _ in range(feature_len)]
                onehot_encoded_feature[feature_int] = 1
                merged_encoding += onehot_encoded_feature

            while len(merged_encoding) < feature_len * maxlen:
                merged_encoding.insert(0, 0)

            X_train.append(merged_encoding)
            outcome = str(df.at[i, 'label'])
            y_train.append(outcome_char_to_int[outcome])

        X_train = np.asarray(X_train)
        y_train = np.asarray(y_train)
        return X_train, y_train

    def one_hot_encoding(self, df, coded_activity, coded_labels, maxlen):
        Outcomes_list = list(coded_labels.keys())
        feature_set = list(coded_activity.keys())
        feature_len = len(feature_set)
        dict_feature_char_to_int = dict((str(c), i) for i, c in enumerate(feature_set))
        outcome_char_to_int = {j: i for i, j in enumerate(Outcomes_list)}

        X_train = list()
        y_train = list()

        for ii in range(0, len(df)):
            onehot_encoded_X = list()
            hist_len = int(df.at[ii, "k"])
            parsed_hist = str(df.at[ii, "prefix"]).split("+")
            for jj in range(hist_len):
                merged_encoding = list()
                feature = parsed_hist[jj]
                feature_int = dict_feature_char_to_int[feature]
                onehot_encoded_feature = [0 for _ in range(feature_len)]
                onehot_encoded_feature[feature_int] = 1
                merged_encoding += onehot_encoded_feature
                onehot_encoded_X.append(merged_encoding)
            while len(onehot_encoded_X) < maxlen:
                onehot_encoded_X.insert(0, [0] * feature_len)
            X_train.append(onehot_encoded_X)

            # prepare y
            outcome = str(df.at[ii, 'label'])
            y_train.append(outcome_char_to_int[outcome])

        X_train = np.asarray(X_train)
        y_train = np.asarray(y_train)
        return X_train, y_train
