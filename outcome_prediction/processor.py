import pandas as pd
import math
import random
import numpy as np
from pm4py.algo.filtering.log.variants import variants_filter
from pm4py.algo.transformation.log_to_features import algorithm as log_to_features
from pm4py.objects.log.obj import EventLog
from pm4py.statistics.traces.generic.log import case_statistics
from pm4py.statistics.variants.log import get as variants_get
from scipy.spatial import distance as sp

random.seed(2021)


class LogsDataProcessor:
    def __init__(self, filepath, dire_path, sampling_feature, pool=1):
        self._dir_path = dire_path
        self._filepath = filepath
        self.sampling_feature = sampling_feature
        self._pool = pool
        self.case_id_col = "case:concept:name"
        self.activity_col = "concept:name"
        self.time_col = "time:timestamp"
        self.label_col = "label"
        self.pos_label = "deviant"
        self.neg_label = "regular"
        self.df = self._load_df()


    def _load_df(self):
        df = pd.read_csv(self._filepath, sep=",")
        df = df[[self.case_id_col, self.activity_col, self.label_col, self.time_col, self.sampling_feature]]
        df.columns = [self.case_id_col, self.activity_col, self.label_col, self.time_col, self.sampling_feature]

        df[self.activity_col] = df[self.activity_col].str.lower()
        df[self.activity_col] = df[self.activity_col].str.replace(" ", "-")

        return df

    def _extract_logs_metadata(self, df):
        activities = list(df[self.activity_col].unique())
        outcomes = [self.pos_label, self.neg_label]
        coded_activity = dict(zip(activities, range(len(activities))))
        coded_labels = dict(zip(outcomes, range(len(outcomes))))
        return coded_activity, coded_labels

    def outcome_prefix_processor(self, df, Max_prefix_length):
        case_id, case_name = self.case_id_col, self.activity_col
        processed_df = pd.DataFrame(columns=[self.case_id_col, "prefix", "k", self.label_col, ])

        idx = 0
        unique_cases = df[case_id].unique()
        for _, case in enumerate(unique_cases):
            act = df[df[case_id] == case][case_name].to_list()
            Outcome = df[df[case_id] == case][self.label_col].to_list()[0]
            if len(act) > Max_prefix_length:
                act = act[:Max_prefix_length]
            for i in range(1, len(act) - 1):
                prefix = np.where(i == 0, act[0], "+".join(act[:i + 1]))
                processed_df.at[idx, self.case_id_col] = case
                processed_df.at[idx, "prefix"] = prefix
                processed_df.at[idx, "k"] = i+1
                processed_df.at[idx, self.label_col] = Outcome
                idx = idx + 1
                # print(idx, Outcome)

        return processed_df

    def _process_outcome(self, df, train_list, test_list, Max_prefix_length):
        processed_df = self.outcome_prefix_processor(df, Max_prefix_length)
        for k in train_list:
            train_df = processed_df[processed_df[self.case_id_col].isin(train_list[k])]
            test_df = processed_df[processed_df[self.case_id_col].isin(test_list[k])]
            train_df.to_csv(f"{self._dir_path}/outcome_train_%s.csv" % k, index=False)
            test_df.to_csv(f"{self._dir_path}/outcome_test_%s.csv" % k , index=False)

    def test_train_spliting(self, df, K_fold):
        train_list = dict()
        test_list = dict()
        Unique_cases = df[self.case_id_col].unique().tolist()
        split_step = len(Unique_cases) / K_fold
        for k in range(K_fold):
            start_range = int(0 + (k * split_step))
            end_range = int(start_range + split_step)
            test_list[k + 1] = Unique_cases[start_range:end_range]
            train_list[k + 1] = list(set(Unique_cases).difference(set(test_list[k + 1])))

        return train_list, test_list

    def process_logs(self,
                     k_fold=5,
                     trunc_ratio=0.90):

        df = self.df
        Max_prefix_length = int(
                np.ceil(df[df[self.label_col] == self.pos_label].groupby(self.case_id_col).size().quantile(trunc_ratio)))
        coded_activity, coded_labels = self._extract_logs_metadata(df)
        train_list, test_list = self.test_train_spliting(df, k_fold)
        # processed_df = self._outcome_helper_func(df, Max_prefix_length)
        # self._process_outcome(df, train_list, test_list, Max_prefix_length)

        return train_list, test_list, Max_prefix_length, coded_activity, coded_labels

    def instanceSelection(self, log, selection_Method, base, feature):
        variants = variants_filter.get_variants(log)
        # all_case_durations = case_statistics.get_all_case_durations(log, parameters={
        #     case_statistics.Parameters.TIMESTAMP_KEY: "time:timestamp"})
        # durations = []
        feature_vectors, feature_names = log_to_features.apply(log, parameters={"str_ev_attr": [],
                                                                                # "concept:name","org:resource"
                                                                                "str_tr_attr": [],
                                                                                "num_ev_attr": [feature],
                                                                                "num_tr_attr": [],
                                                                                "enable_times_from_first_occurrence": True,
                                                                                "enable_case_duration": True,
                                                                                # "str_evsucc_attr": []
                                                                                })
        variant_indexes = variants_get.get_variants_from_log_trace_idx(log)
        print(feature_names)
        # vectors = np.asarray(feature_vectors)
        scores = [0 for i in range(len(log))]
        for i, k in enumerate(variant_indexes):
            if len(variant_indexes[k]) > 2:
                for j in variant_indexes[k]:
                    distance = 0
                    for l in variant_indexes[k]:
                        x = feature_vectors[j]
                        y = feature_vectors[l]
                        distance += sp.euclidean(x, y)
                    scores[j] = distance
                variant_indexes[k] = sorted(variant_indexes[k], key=lambda x: scores[x],
                                            reverse=False)

        pp_log = EventLog()
        pp_log._attributes = log.attributes
        for i, k in enumerate(variants):
            if selection_Method == 'unique':
                pp_log.append(variants[k][0])
            elif selection_Method == 'divide':
                for j in range(math.ceil(len(variants[k]) / base)):
                    pp_log.append(variants[k][j])
            else:  ## log
                for j in range(round(math.log(len(variants[k]), base))):
                    pp_log.append(variants[k][j])

        return pp_log

