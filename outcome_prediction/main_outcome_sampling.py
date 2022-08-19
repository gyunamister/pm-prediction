import argparse
import os
import time
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from xgboost import XGBClassifier

from processor import LogsDataProcessor
import pandas as pd
from pm4py.objects.log.util import dataframe_utils
from pm4py.objects.conversion.log import converter as log_converter
from feature_encoder import FeatureEncoder
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout


def LSTM_model(params):
    # Keras LSTM model
    model = Sequential()
    if params['layers'] == 1:
        model.add(LSTM(units=params['units'], input_shape=(train_X.shape[1], train_X.shape[2])))
        model.add(Dropout(rate=params['rate']))
    else:
        # First layer specifies input_shape and returns sequences
        model.add(
            LSTM(units=params['units'], return_sequences=True, input_shape=(train_X.shape[1], train_X.shape[2])))
        model.add(Dropout(rate=params['rate']))
        # Middle layers return sequences
        for i in range(params['layers'] - 2):
            model.add(LSTM(units=params['units'], return_sequences=True))
            model.add(Dropout(rate=params['rate']))
        # Last layer doesn't return anything
        model.add(LSTM(units=params['units']))
        model.add(Dropout(rate=params['rate']))

    model.add(Dense(1, activation='sigmoid'))
    # model.compile(optimizer='adam', loss='mean_squared_error')
    if params['opt'] == 'adam':
        opt = keras.optimizers.Adam(lr=params['learning_rate'])
    else:
        opt = keras.optimizers.RMSprop(lr=params['learning_rate'])
    model.compile(loss='binary_crossentropy', optimizer=opt,
                  metrics=[tf.keras.metrics.Precision(),
                           tf.keras.metrics.Recall(),'acc'])

    return model


def evaluate(model, x_test, y_test):
    y_pred = model.predict(x_test)

    predictions = [int(np.round(a)) for a in y_pred]
    predictions = np.array(predictions)

    y_test_integer = [int(np.round(a)) for a in y_test]
    y_test_integer = np.array(y_test_integer)

    Accuracy = accuracy_score(y_test_integer, predictions)
    F1_Score = f1_score(y_test_integer, predictions, average='weighted')

    return Accuracy, F1_Score


parser = argparse.ArgumentParser(
    description="LSTM - outcome - Data Processing.")

parser.add_argument("--dataset",
                    type=str,
                    default="bpic2012_ACCEPTED",
                    help="dataset name")

parser.add_argument("--dir_path",
                    type=str,
                    default="./datasets",
                    help="path to store processed data")

parser.add_argument("--raw_log_file",
                    type=str,
                    default="./datasets/BPIC12_acc_Trunc40.csv",
                    help="path to raw csv log file")

parser.add_argument("--model",
                     type=str,
                    default='xgb',
                    help='lstm or xgb')

parser.add_argument('--status',
                    type=str,
                    default='p',
                    help='p: active sampling, o: without sampling')

parser.add_argument('--trunc_ratio',
                    default=1,
                    type=float)

parser.add_argument('--K_fold'
                    , default=5,
                    type=int)

parser.add_argument('--sampling',
                    type=str,
                    default='unique',
                    help="unique or divide or log")

parser.add_argument('--sampling_param',
                    type=int,
                    default=2,
                    help="value for k in sampling methods")

parser.add_argument('--sampling_feature',
                    type=str,
                    default='AMOUNT_REQ',
                    help="feature from event log for sampling")


# Model Params
parser.add_argument('--learning_rate', type=float, default=0.0001)
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--layers', type=int, default=2)
parser.add_argument('--opt', type=str, default='adam', help="RMSprop or adam")
parser.add_argument('--rate', type=float, default=0.2)
parser.add_argument('--units', type=int, default=50)
parser.add_argument('--num_epochs', type=int, default=100)


args = parser.parse_args()

if __name__ == "__main__":

    if not os.path.exists(f"{args.dir_path}/{args.dataset}"):
        os.makedirs(f"{args.dir_path}/{args.dataset}")

    output_datasets_address = f"{args.dir_path}/{args.dataset}"

    print("data preprocessing...")
    start = time.time()
    data_processor = LogsDataProcessor(filepath=args.raw_log_file,
                                       dire_path=output_datasets_address,
                                       sampling_feature=args.sampling_feature)

    train_list, test_list,\
    Max_prefix_length, coded_activity, coded_labels = data_processor.process_logs(k_fold=args.K_fold,
                                                                                  trunc_ratio=args.trunc_ratio)

    Training_data = dict()
    Testing_data = dict()
    for k in range(1, args.K_fold + 1):
        Training_data[k] = data_processor.df[data_processor.df[data_processor.case_id_col].isin(train_list[k])]
        Testing_data[k] = data_processor.df[data_processor.df[data_processor.case_id_col].isin(test_list[k])]
        if args.status == 'p':
            event_log = dataframe_utils.convert_timestamp_columns_in_df(Training_data[k])

            event_log = log_converter.apply(event_log)
            sampled_log = data_processor.instanceSelection(event_log,
                                                           selection_Method=args.sampling,
                                                           base=args.sampling_param,
                                                           feature=args.sampling_feature)

            Training_data[k] = log_converter.apply(sampled_log, variant=log_converter.Variants.TO_DATA_FRAME)

        Training_data[k] = data_processor.outcome_prefix_processor(Training_data[k], Max_prefix_length)
        Testing_data[k] = data_processor.outcome_prefix_processor(Testing_data[k], Max_prefix_length)

    Train_X, Train_y, Test_X, Test_y = dict(), dict(), dict(), dict()
    fe = FeatureEncoder()
    if args.model == 'lstm':
        for k in range(1, args.K_fold + 1):
            Train_X[k], Train_y[k] = fe.one_hot_encoding(Training_data[k] , coded_activity, coded_labels, Max_prefix_length)
            Test_X[k], Test_y[k] = fe.one_hot_encoding(Testing_data[k], coded_activity, coded_labels, Max_prefix_length)
    elif args.model == 'xgb':
        for k in range(1, args.K_fold + 1):
            Train_X[k], Train_y[k] = fe.one_hot_encoding_xgb(Training_data[k] , coded_activity, coded_labels, Max_prefix_length)
            Test_X[k], Test_y[k] = fe.one_hot_encoding_xgb(Testing_data[k], coded_activity, coded_labels, Max_prefix_length)

    Feature_Extraction_Time = time.time() - start
    print(f"Average Feature Extraction time: {Feature_Extraction_Time / args.K_fold}")

########################################################################################################################
    print("Training model ...")
    params = {'batch_size': args.batch_size, 'layers': args.layers, 'learning_rate': args.learning_rate,
              'opt': args.opt, 'rate': args.rate, 'units': args.units}
    ACC = []
    F1 = []
    Training_time = []
    if args.model == 'lstm':
        for k in range(1, args.K_fold + 1):
            start_time = time.time()
            train_X, train_y = Train_X[k], Train_y[k]
            model = LSTM_model(params)
            early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, mode='auto')
            lr_reducer = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, verbose=0,
                                                           mode='auto', min_delta=0.0001, cooldown=0, min_lr=0)
            model.fit(train_X, train_y , batch_size=params['batch_size'], epochs=args.num_epochs, verbose=1,
                      validation_split=0.2, callbacks=[early_stopping, lr_reducer])

            Running_Time = time.time() - start_time
            Training_time.append(Running_Time)
            print('LSTM training time for fold %s is %.3f' % (k, Running_Time))

            Accuracy, F1_Score = evaluate(model, Test_X[k], Test_y[k])
            ACC.append(Accuracy)
            F1.append(F1_Score)
            print("acuracy for fold = %s : %.5f" % (k, Accuracy))
            print("F1-score for fold = %s : %.5f" % (k, F1_Score))

    elif args.model == 'xgb':
        for k in range(1, args.K_fold + 1):
            start_time = time.time()
            train_X, train_y = Train_X[k], Train_y[k]
            model = XGBClassifier()
            model.fit(train_X, train_y, )
            Running_Time = time.time() - start_time
            Training_time.append(Running_Time)
            print('XGB training time for fold %s is %.3f' % (k, Running_Time))

            y_prediction = model.predict(Test_X[k])
            predictions = [round(value) for value in y_prediction]
            Accuracy = accuracy_score(Test_y[k], predictions)
            F1_Score = f1_score(Test_y[k], predictions, average="weighted")

            ACC.append(Accuracy)
            F1.append(F1_Score)
            print("acuracy for fold %s : %.5f" % (k, Accuracy))
            print("F1-score for fold %s : %.5f" % (k, F1_Score))

    results_df = pd.DataFrame({"average accuracy": [np.mean(ACC)], "average F1-score": [np.mean(F1)],
                               "average time": [np.mean(Training_time)]})

    final_K_folds_results_name = output_datasets_address + "/FinalResluts_%s_%s_K%s_%s.csv" % (args.dataset,
                                                                                              args.sampling,
                                                                                              args.sampling_param,
                                                                                              args.model)
    results_df.to_csv(final_K_folds_results_name, index=False)

