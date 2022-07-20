import traceback
import pickle
import os.path
import pred_config
from core.feature_encoder import FeatureEncoder
from core.feature_generator import FeatureGenerator
from core.model import net
import time
import numpy as np
import csv

args = ""

if __name__ == '__main__':
    args = pred_config.load()
    
    task = args.task
    
    contextual_info = args.contextual_info
    control_flow_p = args.control_flow_p
    time_p = args.time_p
    resource_p = args.resource_p
    data_p = args.data_p
    transition = args.transition
    
    num_epochs = args.num_epochs
    batch_size = args.batch_size
    
    data_dir = args.data_dir
    data_set = args.data_set

    checkpoint_dir = args.checkpoint_dir
    
    test_set_dir = args.test_data_dir
    test_data_set = args.test_data_set

    cross_number = args.cross_number

    exp_name = '%s_%s_%s_%s_%s_%s_%s_%s' % (data_set, task, control_flow_p, time_p, resource_p, data_p, num_epochs, batch_size)

    total_results_file = args.result_dir + exp_name + '.csv'

    file_exists = os.path.isfile(total_results_file)

    load_saved_data = args.load_saved_data
    load_saved_test_data = args.load_saved_test_data 
    save_data = args.save_data

    with open (total_results_file, 'a') as csvfile:
        headers = ['Dataset', 'Cross Validation', 'Accuracy', 'F1_Score', 'Precisions', 'Recalls', 'Running_Time', 'FeatureEncoder_Time',
              'TruePositive', 'FalsePositive', 'FalseNegative', 'TrueNegative']
        writer = csv.DictWriter(csvfile, delimiter=',', lineterminator='\n',fieldnames=headers)

        if not file_exists:
            writer.writeheader()  # file doesn't exist yet, write a header

    for i in range(cross_number):
        try:
            print("Cross number %d: " % (i+1))
            filename = data_dir + data_set + "_%d.csv" % (i+1)
            model_name = '%s_%s_%s_%s_%s_%s_%s_%s' % (data_set, task, control_flow_p, time_p, resource_p, data_p, num_epochs,
                batch_size)

            feature_name = '%s_%s_%s_%s_%s_%s' % (
                    data_set, task, control_flow_p, time_p, resource_p, data_p)

            log_config = {"control_flow_p": control_flow_p, "time_p": time_p, "resource_p": resource_p, "data_p": data_p, 'transition': transition}

            if load_saved_data:
                ## Load Saved Datasets
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
                ## load data
                print("flag: loading training data")
                Start_FeatureGenerator = time.time()
                fg = FeatureGenerator()
                df = fg.create_initial_log(filename, log_config)
                FeatureGenerator_Time = time.time() - Start_FeatureGenerator
                print("FeatureGenerator done")

                num_events = len(df)
                num_cases = len(set(df["id"]))

                # feature generation
                print("flag: generating features")
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
                    df = fg.add_next_activity(df)
                    df = fg.add_query_remaining(df)
                print("done")

                # training set generation
                print("flag: encoding features")
                Start_FeatureEncoder = time.time()
                fe = FeatureEncoder()
                train_X, train_y = fe.original_one_hot_encode(df, feature_type_list, task, feature_name)
                FeatureEncoder_Time = time.time() - Start_FeatureEncoder
                print("done")

                if save_data:
                    with open("%s/%s-%s.pkl" % (data_dir, feature_name, "X"), 'wb') as f:
                        pickle.dump(train_X, f)
                    with open("%s/%s-%s.pkl" % (data_dir, feature_name, "y"), 'wb') as f:
                        pickle.dump(train_y, f)
                    print("flag: training data is saved")

            print("flag: training model")
            if contextual_info:
                train_df = fg.queue_level(train_df)
                activity_list = fg.get_activities(train_df)
                train_context_X = fg.generate_context_feature(train_df, activity_list)
                model = net()
                if task == 'next_timestamp':
                    model.train(train_X, train_y, regression, loss, n_epochs=num_epochs, batch_size=batch_size,
                                model_name=model_name, checkpoint_dir=checkpoint_dir,
                                X_train_ctx=train_context_X)
                elif task == 'next_activity':
                    model.train(train_X, train_y, regression, loss, n_epochs=num_epochs, batch_size=batch_size,
                                model_name=model_name, checkpoint_dir=checkpoint_dir,
                                X_train_ctx=train_context_X)
            else:
                train_context_X = None
                model = net()
                if task == 'next_timestamp':
                    model.train(train_X, train_y, regression, loss, n_epochs=num_epochs, batch_size=batch_size,
                                model_name=model_name, checkpoint_dir=checkpoint_dir,
                                context=contextual_info)
                elif task == 'next_activity':
                    model.train(train_X, train_y, regression, loss, n_epochs=num_epochs, batch_size=batch_size,
                                model_name=model_name, checkpoint_dir=checkpoint_dir,
                                context=contextual_info)

            Running_Time = model.running_time
            print("training is done")
            model.load(checkpoint_dir, model_name=model_name)

            if load_saved_test_data:
                with open("%s/%s-%s.pkl" % (test_set_dir, feature_name, "X"), "rb") as text_X_file:
                    test_X = pickle.load(text_X_file)
                with open("%s/%s-%s.pkl" % (test_set_dir, feature_name, "y"), "rb") as text_y_file:
                    test_y = pickle.load(text_y_file)
                print("flag: test data is loaded")

            else:
                ## loading test data
                filename = test_set_dir + test_data_set + "_%d.csv" % (i+1)
                print("flag: loading test data")
                print('file name is: ' + filename)
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
                test_X, test_y = fe.preprocessed_one_hot_encode(df, feature_type_list, task, feature_name)
                print("done")

                if save_data:
                    with open("%s/test_%s-%s.pkl" % (test_set_dir, feature_name, "X"), 'wb') as f:
                        pickle.dump(test_X, f)
                    with open("%s/test_%s-%s.pkl" % (test_set_dir, feature_name, "y"), 'wb') as f:
                        pickle.dump(test_y, f)
                    print("flag: test set is saved")

            exp_info = {"task": task, "filename": filename, "control_flow_p": control_flow_p, "time_p": time_p,
                        "resource_p": resource_p, "data_p": data_p, "num_epochs": num_epochs,
                        "batch_size": batch_size}

            # Evaluate the model on the test data using `evaluate`
            CF_matrix, report, Accuracy, F1_Score, Precision, Recall = model.evaluate(test_X, test_y, exp_info)

            TruePositive = sum(np.diag(CF_matrix))
            FalsePositive = 0
            for jj in range(CF_matrix.shape[0]):
                FalsePositive += sum(CF_matrix[:, jj]) - CF_matrix[jj, jj]
            FalseNegative = 0
            for ii in range(CF_matrix.shape[0]):
                FalseNegative += sum(CF_matrix[ii, :]) - CF_matrix[ii, ii]
            TrueNegative = 0
            for kk in range(CF_matrix.shape[0]):
                temp = np.delete(CF_matrix, kk, 0)
                temp = np.delete(temp, kk, 1)
                TrueNegative += sum(sum(temp))

            Results_data = [data_set, (i+1), Accuracy, F1_Score, Precision, Recall, Running_Time, FeatureGenerator_Time,
                            TruePositive, FalsePositive, FalseNegative, TrueNegative]

            print("flag: saving output")
            with open(total_results_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(Results_data)
            print("done")
        except Exception as e:
            traceback.print_exc()
            continue

    print(f"done all {cross_number} validation")
