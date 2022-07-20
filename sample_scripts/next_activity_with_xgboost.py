import os
import pickle
import time
import pred_config
import csv
from core.feature_encoder import FeatureEncoder
from core.feature_generator import FeatureGenerator
from xgboost import XGBClassifier
from sklearn.metrics import f1_score, accuracy_score, recall_score, \
    precision_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
import traceback

if __name__ == '__main__':
    args = pred_config.load()

    status = args.status
    task = args.task
    control_flow_p = args.control_flow_p
    time_p = args.time_p
    resource_p = args.resource_p
    data_p = args.data_p
    transition = args.transition

    data_dir = args.data_dir
    data_set = args.data_set
    
    checkpoint_dir = args.checkpoint_dir

    test_data_dir = args.test_data_dir
    test_data_set = args.test_data_set
    
    cross_number = args.cross_number

    exp_name = '%s_%s_%s_%s_%s_%s_XGB' % (data_set, task, control_flow_p, time_p, resource_p, data_p)

    load_saved_data = args.load_saved_data
    load_saved_test_data = args.load_saved_test_data
    save_data = args.save_data

    total_results_file = args.result_dir + exp_name + '.csv'
    file_exists = os.path.isfile(total_results_file)

    with open(total_results_file, 'a', newline='') as csvfile:
        headers = ['Dataset', 'Cross Validation', 'Accuracy', 'F1_Score', 'Precisions', 'Recalls', 'Running_Time', 'FeatureEncoder_Time',
              'TruePositive', 'FalsePositive', 'FalseNegative', 'TrueNegative']
        writer = csv.DictWriter(csvfile, delimiter=',', lineterminator='\n',fieldnames=headers)
        if not file_exists:
            writer.writeheader()  # file doesn't exist yet, write a header
            
    for i in range(cross_number):
        try:
            print("Cross number %d: " % i)
            filename = data_dir + data_set + "_%d.csv" % (i+1)
            model_name = '%s_%s_%s_%s_%s_%s_%s_XGB' % (
                data_set, task, control_flow_p, time_p, resource_p, data_p, transition)
            feature_name = '%s_%s_%s_%s_%s_%s_%s_XGB' % (
                data_set, task, control_flow_p, time_p, resource_p, data_p, transition)

            log_config = {"control_flow_p": control_flow_p, "time_p": time_p, "resource_p": resource_p, "data_p": data_p, "transition": transition}

            # load data
            if load_saved_data:
                # Load Saved Datasets
                with open("%s/%s-%s.pkl" % (data_dir, feature_name, "X"), "rb") as text_X_file:
                    train_X = pickle.load(text_X_file)
                with open("%s/%s-%s.pkl" % (data_dir, feature_name, "y"), "rb") as text_y_file:
                    train_y = pickle.load(text_y_file)

                print("flag: data is loaded")

                if task == 'next_activity':
                    loss = 'categorical_crossentropy'
                    regression = False
                    feature_type_list = ["activity_history"]

                elif task == 'next_timestamp':
                    loss = 'mae'
                    regression = True
                    feature_type_list = ["activity_history"]

            else:
                print("flag: loading training data")
                fg = FeatureGenerator()
                df = fg.create_initial_log(filename, log_config)
                print("done")

                num_events = len(df)
                num_cases = len(set(df["id"]))

                # feature generation
                print("flag: generating train features")
                if task == 'next_activity':
                    loss = 'categorical_crossentropy'
                    regression = False
                    feature_type_list = ["activity_history"]
                    df = fg.add_activity_history(df)
                    df = fg.add_next_activity(df)
                elif task == 'next_timestamp':
                    loss = 'mae'
                    regression = True
                    feature_type_list = ["activity_history"]
                    df = fg.add_activity_history(df)
                    df = fg.add_query_remaining(df)
                print("done")

                # # training set generation
                print("flag: encoding train features")
                Start_FeatureEncoder = time.time()
                fe = FeatureEncoder()
                print("start encoding original data")
                train_X, train_y = fe.original_one_hot_encode_xgb(df, feature_type_list, task, feature_name)
                FeatureEncoder_Time = time.time() - Start_FeatureEncoder
                print("done")

            print("flag: training model")
            train_X, X_validation, train_y, y_validation = train_test_split(train_X, train_y, test_size=0.1)
            model = XGBClassifier()
            Start_time = time.time()
            early_stopping = 10
            model.fit(train_X, train_y,
                    eval_set=[(train_X, train_y), (X_validation, y_validation)],
                    early_stopping_rounds=20
                    )
            training_time = time.time() - Start_time
            print('training time is %.3f S' % training_time)

            if load_saved_test_data:
                with open("%s/%s-%s.pkl" % (test_data_dir, feature_name, "X"), "rb") as text_X_file:
                    test_X = pickle.load(text_X_file)
                with open("%s/%s-%s.pkl" % (test_data_dir, feature_name, "y"), "rb") as text_y_file:
                    test_y = pickle.load(text_y_file)
            else:
                # loading test data
                filename = test_data_dir + test_data_set + "_%d.csv" % (i+1)
                print("flag: loading test data")
                fg = FeatureGenerator()
                df = fg.create_initial_log(filename, log_config)
                print("done")
                num_events = len(df)
                num_cases = len(set(df["id"]))
                # feature generation
                print("flag: generating features")
                if task == 'next_activity':
                    df = fg.add_activity_history(df)
                    df = fg.add_next_activity(df)
                elif task == 'next_timestamp':
                    df = fg.add_activity_history(df)
                    df = fg.add_query_remaining(df)
                print("done")
                # training set generation
                print("flag: encoding features")
                fe = FeatureEncoder()
                test_X, test_y = fe.preprocessed_one_hot_encode_xgb(df, feature_type_list, task, feature_name)
                print("done")

                if save_data:
                    with open("%s/test_%s-%s.pkl" % (test_data_dir, feature_name, "X"), 'wb') as f:
                        pickle.dump(test_X, f)
                    with open("%s/test_%s-%s.pkl" % (test_data_dir, feature_name, "y"), 'wb') as f:
                        pickle.dump(test_y, f)
                    print("flag: test set is saved")

            y_pred = model.predict(test_X)
            predictions = [round(value) for value in y_pred]
            # evaluate 
            print("Step1")
            print(confusion_matrix(test_y, predictions))
            TP = 0
            FP = 0
            TN = 0
            FN = 0
            for jj in set(test_y):
                for ii in range(len(predictions)): 
                    if test_y[ii]==predictions[ii]:
                        if test_y[ii]==jj:
                            TP += 1
                        else:
                            TN+=1
                    elif predictions[ii]==jj:
                        FP += 1
                    else: 
                        FN += 1
            print("Step2")
            # Print the precision and recall, among other metrics
            print(classification_report(test_y, predictions, digits=3))
            print("Accuracy: %.5f" % (accuracy_score(test_y, predictions)))
            print("F1-Score: %.5f" % f1_score(test_y, predictions, average="weighted"))
            print("precision: %.5f" % precision_score(test_y, predictions, average="weighted"))
            print("recall: %.5f" % recall_score(test_y, predictions, average="weighted"))

            # with open(total_results_file, 'w') as f:
            #     f.write("Running time is %.5f \n" % training_time)
            #     if not load_saved_data:
            #         f.write("Feature Encoder time is %.5f \n" % FeatureEncoder_Time)
            #     f.write(str(classification_report(test_y, predictions, digits=5)))
            #     f.write("\n")
            #     f.write("Accuracy: %.5f \n" % (accuracy_score(test_y, predictions)))
            #     f.write("F1-Score: %.5f \n" % f1_score(test_y, predictions, average="weighted"))
            #     f.write("precision: %.5f \n" % precision_score(test_y, predictions, average="weighted"))
            #     f.write("recall: %.5f \n" % recall_score(test_y, predictions, average="weighted"))
            #     f.close()
            Accuracy= accuracy_score(test_y, predictions)
            F1_Score= f1_score(test_y, predictions, average="weighted")
            Precision= precision_score(test_y, predictions, average="weighted")
            Recall= recall_score(test_y, predictions, average="weighted")
            Running_Time= training_time
            FeatureGenerator_Time= FeatureEncoder_Time
            TruePositive, FalsePositive, FalseNegative, TrueNegative = TP,FP,FN,TN
            print("Step3")
            # Results_data = [i, Accuracy, F1_Score, Precision, Recall, Running_Time, FeatureGenerator_Time,
            #                     TruePositive, FalsePositive, FalseNegative, TrueNegative]
            Results_data = [data_set, (i+1), Accuracy, F1_Score, Precision, Recall, Running_Time, FeatureGenerator_Time,
                            TruePositive, FalsePositive, FalseNegative, TrueNegative]
            # Saved_data = []

            # with open(total_results_file, 'r') as f:
            #     reader = csv.reader(f)
            #     for row in reader:
            #         Saved_data.append(row)
            # with open(total_results_file, 'w', newline='') as f:
            #     writer = csv.writer(f)
            #     for row in Saved_data:
            #         writer.writerow(row)
            #     writer.writerow(Results_data)
            print("flag: saving output")
            with open(total_results_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(Results_data)
            print("done")
        except Exception as e:
            traceback.print_exc()
            continue

    print("done all 5 validation")

    # print(predictions)
    # print("**********")
    # print(test_y)
    # recall = recall_score(test_y, predictions)
    # precision = precision_score(test_y, predictions)
    # print("Accuracy: %.2f%% , precision: %.2f%% , recall: %.2f%%" % (accuracy, precision, recall))
