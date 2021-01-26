import os
import sys
from datetime import datetime as d_time
import random

import pickle
import numpy as np
import pandas as pd

class FeatureEncoder(object):
  # todos Separate the encoding for activity histroy, resource history, remaining time, service time, etc.
  # todos Separate x and y
  def original_one_hot_encode_xgb(self, df, feature_type_list, target_feature,feature_name):
    dict_dir="../feature_infos/Train_All"
    num_features_dict = dict()
    feature_len = 0 
    dict_feature_char_to_int = dict()
    for feature_type in feature_type_list:
      feature_set_name = feature_type.split("_")[0]
      print(feature_set_name)
      feature_set = sorted(list(set(df[feature_set_name])))
      if feature_type=="activity_history":
        feature_set.append('!')
      with open("%s/%s_%s.pkl" % (dict_dir,feature_type,feature_name), 'wb') as f:
        pickle.dump(feature_set, f)
      num_feature = len(feature_set)
      feature_len += num_feature
      num_features_dict[feature_type] = num_feature
      dict_feature_char_to_int[feature_type] = dict((str(c), i) for i, c in enumerate(feature_set))
      feature_int_to_char = dict((i, c) for i, c in enumerate(feature_set))

    X_train = list()
    y_train = list()
    maxlen = max([len(str(x).split('_')) for x in df[feature_type_list[0]]])
    with open("%s/%s_%s.pkl" % (dict_dir,"maxlen",feature_name), 'wb') as f:
        pickle.dump(maxlen, f)
    for i in range(0, len(df)):
      # print("{}th among {}".format(i,len(df)))
      # prepare X
      onehot_encoded_X = list()
      hist_len = len(str(df.at[i, feature_type_list[0]]).split("_"))
      merged_encoding =list()
      for j in range(hist_len):
        # merged_encoding =list()
        for feature_type in feature_type_list:
          parsed_hist = str(df.at[i, feature_type]).split("_")
          feature = parsed_hist[j]
          feature_int = dict_feature_char_to_int[feature_type][feature]
          onehot_encoded_feature = [0 for _ in range(num_features_dict[feature_type])]
          onehot_encoded_feature[feature_int] = 1
          merged_encoding += onehot_encoded_feature
        # onehot_encoded_X.append(merged_encoding)
      while len(merged_encoding) != maxlen * num_feature:
        merged_encoding.insert(0, 0)
      # print(merged_encoding)
      X_train.append(merged_encoding)

      # merged_encoding =list()
      # for feature_type in feature_type_list:
      #   if df.at[i, feature_type] == "nan":
      #     continue
      #   else:  
      #     parsed_hist = str(df.at[i, feature_type]).split("_")
      #     int_encoded_feature = [dict_feature_char_to_int[feature_type][feature] for feature in parsed_hist]
      #     for feature_int in int_encoded_feature:
      #       onehot_encoded_feature = [0 for _ in range(num_features_dict[feature_type])]
      #       onehot_encoded_feature[feature_int] = 1
      #       merged_encoding += onehot_encoded_feature
      # while len(merged_encoding) < maxlen:
      #   merged_encoding.insert(0, [0]*(feature_len))
      # X_train.append(merged_encoding)


      # prepare y
      if target_feature == "next_activity":
        next_act = str(df.at[i, 'next_activity'])
        print(dict_feature_char_to_int)
        int_encoded_next_act = dict_feature_char_to_int["activity_history"][next_act]
        activities = sorted(list(set(df['activity'])))
        activities.append("!")
        # onehot_encoded_next_act = [0 for _ in range(len(activities))]
        # onehot_encoded_next_act[int_encoded_next_act] = 1
        y_train.append(int_encoded_next_act)
      elif target_feature == "next_timestamp":
        remaining_time = df.at[i, 'remaining_time']
        y_train.append(remaining_time)

    X_train = np.asarray(X_train)
    y_train = np.asarray(y_train)
    return X_train, y_train
  
  def original_one_hot_encode(self, df, feature_type_list, target_feature,feature_name):
    dict_dir="../feature_infos/Train_All"
    num_features_dict = dict()
    feature_len = 0 
    dict_feature_char_to_int = dict()
    for feature_type in feature_type_list:
      feature_set_name = feature_type.split("_")[0]
      print(feature_set_name)
      feature_set = sorted(list(set(df[feature_set_name])))
      if feature_type=="activity_history":
        feature_set.append('!')
      with open("%s/%s_%s.pkl" % (dict_dir,feature_type,feature_name), 'wb') as f:
        pickle.dump(feature_set, f)
      num_feature = len(feature_set)
      feature_len += num_feature
      num_features_dict[feature_type] = num_feature
      dict_feature_char_to_int[feature_type] = dict((str(c), i) for i, c in enumerate(feature_set))
      feature_int_to_char = dict((i, c) for i, c in enumerate(feature_set))

    X_train = list()
    y_train = list()
    maxlen = max([len(str(x).split('_')) for x in df[feature_type_list[0]]])
    with open("%s/%s_%s.pkl" % (dict_dir,"maxlen",feature_name), 'wb') as f:
        pickle.dump(maxlen, f)
    for i in range(0, len(df)):
      # print("{}th among {}".format(i,len(df)))
      # prepare X
      onehot_encoded_X = list()
      hist_len = len(str(df.at[i, feature_type_list[0]]).split("_"))
      for j in range(hist_len):
        merged_encoding =list()
        for feature_type in feature_type_list:
          parsed_hist = str(df.at[i, feature_type]).split("_")
          feature = parsed_hist[j]
          feature_int = dict_feature_char_to_int[feature_type][feature]
          onehot_encoded_feature = [0 for _ in range(num_features_dict[feature_type])]
          onehot_encoded_feature[feature_int] = 1
          merged_encoding += onehot_encoded_feature
        onehot_encoded_X.append(merged_encoding)
      while len(onehot_encoded_X) != maxlen:
        onehot_encoded_X.insert(0, [0]*(feature_len))
      X_train.append(onehot_encoded_X)

      # merged_encoding =list()
      # for feature_type in feature_type_list:
      #   if df.at[i, feature_type] == "nan":
      #     continue
      #   else:  
      #     parsed_hist = str(df.at[i, feature_type]).split("_")
      #     int_encoded_feature = [dict_feature_char_to_int[feature_type][feature] for feature in parsed_hist]
      #     for feature_int in int_encoded_feature:
      #       onehot_encoded_feature = [0 for _ in range(num_features_dict[feature_type])]
      #       onehot_encoded_feature[feature_int] = 1
      #       merged_encoding += onehot_encoded_feature
      # while len(merged_encoding) < maxlen:
      #   merged_encoding.insert(0, [0]*(feature_len))
      # X_train.append(merged_encoding)


      # prepare y
      if target_feature == "next_activity":
        next_act = str(df.at[i, 'next_activity'])
        int_encoded_next_act = dict_feature_char_to_int["activity_history"][next_act]
        activities = sorted(list(set(df['activity'])))
        activities.append("!")
        onehot_encoded_next_act = [0 for _ in range(len(activities))]
        onehot_encoded_next_act[int_encoded_next_act] = 1
        y_train.append(onehot_encoded_next_act)
      elif target_feature == "next_timestamp":
        remaining_time = df.at[i, 'remaining_time']
        y_train.append(remaining_time)

    X_train = np.asarray(X_train)
    y_train = np.asarray(y_train)
    return X_train, y_train

  def preprocessed_one_hot_encode(self, df, feature_type_list, target_feature,feature_name):
    dict_dir = "../feature_infos/Train_All"
    num_features_dict = dict()
    feature_len = 0
    dict_feature_char_to_int = dict()
    for feature_type in feature_type_list:
        with open("%s/%s_%s.pkl" % (dict_dir, feature_type, feature_name), 'rb') as f:
            feature_set = pickle.load(f)
        num_feature = len(feature_set)
        feature_len += num_feature
        num_features_dict[feature_type] = num_feature
        dict_feature_char_to_int[feature_type] = dict((str(c), i) for i, c in enumerate(feature_set))
        feature_int_to_char = dict((i, c) for i, c in enumerate(feature_set))

    X_train = list()
    y_train = list()
    with open("%s/%s_%s.pkl" % (dict_dir, "maxlen", feature_name), 'rb') as f:
        maxlen = pickle.load(f)
        print(maxlen)
    print(len(df))
    for i in range(0, len(df)):
        # print("{}th among {}".format(i,len(df)))
        # prepare X
      onehot_encoded_X = list()
      hist_len = len(str(df.at[i, feature_type_list[0]]).split("_"))
      for j in range(min(hist_len, maxlen)):
          merged_encoding = list()
          for feature_type in feature_type_list:
              parsed_hist = str(df.at[i, feature_type]).split("_")
              feature = parsed_hist[j]
              feature_int = dict_feature_char_to_int[feature_type][feature]
              onehot_encoded_feature = [0 for _ in range(num_features_dict[feature_type])]
              onehot_encoded_feature[feature_int] = 1
              merged_encoding += onehot_encoded_feature
          onehot_encoded_X.append(merged_encoding)
      while len(onehot_encoded_X) != maxlen:
          onehot_encoded_X.insert(0, [0] * feature_len)
      X_train.append(onehot_encoded_X)
      # prepare y
      if target_feature == "next_activity":
          next_act = str(df.at[i, 'next_activity'])
          int_encoded_next_act = dict_feature_char_to_int["activity_history"][next_act]
          activities = sorted(list(set(df['activity'])))
          activities.append("!")
          # onehot_encoded_next_act = [0 for _ in range(len(activities))]
          onehot_encoded_next_act = [0 for _ in range(feature_len)]
          onehot_encoded_next_act[int_encoded_next_act] = 1
          y_train.append(onehot_encoded_next_act)
      elif target_feature == "next_timestamp":
          remaining_time = df.at[i, 'remaining_time']
          y_train.append(remaining_time)

    X_train = np.asarray(X_train)
    y_train = np.asarray(y_train)
    return X_train, y_train

  def preprocessed_one_hot_encode_xgb(self, df, feature_type_list, target_feature,feature_name):
      dict_dir = "../feature_infos/Train_All"
      num_features_dict = dict()
      feature_len = 0
      dict_feature_char_to_int = dict()
      for feature_type in feature_type_list:
          with open("%s/%s_%s.pkl" % (dict_dir, feature_type, feature_name), 'rb') as f:
              feature_set = pickle.load(f)
          num_feature = len(feature_set)
          feature_len += num_feature
          num_features_dict[feature_type] = num_feature
          dict_feature_char_to_int[feature_type] = dict((str(c), i) for i, c in enumerate(feature_set))
          feature_int_to_char = dict((i, c) for i, c in enumerate(feature_set))

      X_train = list()
      y_train = list()
      with open("%s/%s_%s.pkl" % (dict_dir, "maxlen", feature_name), 'rb') as f:
          maxlen = pickle.load(f)
      for i in range(0, len(df)):
          # print("{}th among {}".format(i,len(df)))
          # prepare X
        onehot_encoded_X = list()
        hist_len = len(str(df.at[i, feature_type_list[0]]).split("_"))
        merged_encoding = list()
        for j in range(min(hist_len, maxlen)):
            # merged_encoding = list()
            for feature_type in feature_type_list:
                parsed_hist = str(df.at[i, feature_type]).split("_")
                feature = parsed_hist[j]
                feature_int = dict_feature_char_to_int[feature_type][feature]
                onehot_encoded_feature = [0 for _ in range(num_features_dict[feature_type])]
                onehot_encoded_feature[feature_int] = 1
                merged_encoding += onehot_encoded_feature
            # onehot_encoded_X.append(merged_encoding)
        while len(merged_encoding) != maxlen * num_feature:
            merged_encoding.insert(0, 0)
        X_train.append(merged_encoding)
        # prepare y
        if target_feature == "next_activity":
            next_act = str(df.at[i, 'next_activity'])
            int_encoded_next_act = dict_feature_char_to_int["activity_history"][next_act]
            activities = sorted(list(set(df['activity'])))
            activities.append("!")
            # onehot_encoded_next_act = [0 for _ in range(len(activities))]
            # onehot_encoded_next_act[int_encoded_next_act] = 1
            y_train.append(int_encoded_next_act)
        elif target_feature == "next_timestamp":
            remaining_time = df.at[i, 'remaining_time']
            y_train.append(remaining_time)

      X_train = np.asarray(X_train)
      y_train = np.asarray(y_train)
      return X_train, y_train