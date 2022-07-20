import pickle
import Cross_config
from feature_encoder import FeatureEncoder
from feature_generator import FeatureGenerator
from model import net
from sklearn.model_selection import train_test_split
import time
import numpy as np
import csv

accuracy_values = list()
accuracy_sum = 0.0
accuracy_value = 0.0
precision_values = list()
precision_sum = 0.0
precision_value = 0.0
recall_values = list()
recall_sum = 0.0
recall_value = 0.0
f1_values = list()
f1_sum = 0.0
f1_value = 0.0
training_time_seconds = list()

args = ""

if __name__ == '__main__':
    args = Cross_config.load()

    status = args.status

    level = args.inter_case_level
    # filename = req['name']
    task = args.task
    contextual_info = args.contextual_info
    num_epochs = args.num_epochs
    batch_size = args.batch_size
    checkpoint_dir = args.checkpoint_dir
    control_flow_p = args.control_flow_p
    time_p = args.time_p
    resource_p = args.resource_p
    transition = args.transition
    data_p = args.data_p
    test_set_dir = args.test_data_dir
    test_data_set = args.test_data_set

    cross_number = args.cross_number

    data_dir = args.data_dir
    data_set = args.data_set
    p_data_set = args.p_data_set

    # if status == 'o':
    #     total_results_file = checkpoint_dir + "Total_Results_" + folder_name + ".csv"
    # elif status == 'p':
    #     total_results_file = checkpoint_dir + "Total_Results_" + p_folder_name + ".csv"
    total_results_file = "../result/new_exp_result.csv"

    import os.path


    file_exists = os.path.isfile(total_results_file)

    with open (total_results_file, 'a') as csvfile:
        headers = ['Dataset', 'Cross Validation', 'Accuracy', 'F1_Score', 'Precisions', 'Recalls', 'Running_Time', 'FeatureEncoder_Time',
              'TruePositive', 'FalsePositive', 'FalseNegative', 'TrueNegative']
        writer = csv.DictWriter(csvfile, delimiter=',', lineterminator='\n',fieldnames=headers)

        if not file_exists:
            writer.writeheader()  # file doesn't exist yet, write a header

    for i in range(cross_number):
        print("Cross number %d: " % (i+2))
        # results_file_name = folder_name[:index + 6] + "%d_" % i + folder_name[index + 6:]
        # p_results_file_name = p_folder_name[:index_p + 6] + "%d_" % i + p_folder_name[index_p + 6:]

        # if status == 'o':
        #     results_file = checkpoint_dir + "Results_" + results_file_name + "%d.txt" % i
        # elif status == 'p':
        #     results_file = checkpoint_dir + "Results_" + p_results_file_name + "%d.txt" % i

        if status == 'o':
            exp_dataset = data_set
            filename = data_dir + data_set + "_%d.csv" % (i+2)
            print(filename)
            model_name = '%s-%s_%s_%s_%s_%s_%s_%s_%s_%s' % (
                status, data_set, args.task, control_flow_p, time_p, resource_p, data_p, transition, num_epochs,
                batch_size)

        elif status == 'p':
            exp_dataset = p_data_set
            filename = data_dir + p_data_set + "_%d.csv" % (i+2)
            model_name = '%s-%s_%s_%s_%s_%s_%s_%s_%s_%s' % (
                status, p_data_set, args.task, control_flow_p, time_p, resource_p, data_p, transition, num_epochs,
                batch_size)

        feature_name = '%s_%s_%s_%s_%s_%s_%s' % (
                data_set, args.task, control_flow_p, time_p, resource_p, data_p, transition)

        log_config = {"control_flow_p": control_flow_p, "time_p": time_p, "resource_p": resource_p, "data_p": data_p,
                      "transition": transition}

        if args.load_saved_data:
            if status == 'o':
                ## Load Saved Datasets
                with open("%s/%s-%s.pkl" % (data_dir, feature_name, "X"), "rb") as text_X_file:
                    train_X = pickle.load(text_X_file)
                with open("%s/%s-%s.pkl" % (data_dir, feature_name, "y"), "rb") as text_y_file:
                    train_y = pickle.load(text_y_file)
            elif status == 'p':
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
            print(FeatureGenerator_Time)
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

            df.to_csv('./training_set_with_features.csv')
            print("done")

            # training set generation
            print("flag: encoding features")
            Start_FeatureEncoder = time.time()
            fe = FeatureEncoder()
            if status == 'o':
                train_X, train_y = fe.original_one_hot_encode(df, feature_type_list, task, feature_name)
            elif status == 'p':
                print(feature_name)
                train_X, train_y = fe.preprocessed_one_hot_encode(df, feature_type_list, task, feature_name)
            else:
                print("status not defined!")

            FeatureEncoder_Time = time.time() - Start_FeatureEncoder
            print("done")

            if args.save_data:
                if status == 'o':
                    with open("%s/%s-%s.pkl" % (data_dir, feature_name, "X"), 'wb') as f:
                        pickle.dump(train_X, f)
                    with open("%s/%s-%s.pkl" % (data_dir, feature_name, "y"), 'wb') as f:
                        pickle.dump(train_y, f)
                elif status == 'p':
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
                            model_name=model_name, checkpoint_dir=args.checkpoint_dir,
                            X_train_ctx=train_context_X)
            elif task == 'next_activity':
                model.train(train_X, train_y, regression, loss, n_epochs=num_epochs, batch_size=batch_size,
                            model_name=model_name, checkpoint_dir=args.checkpoint_dir,
                            X_train_ctx=train_context_X)
        else:
            train_context_X = None
            model = net()
            if task == 'next_timestamp':
                model.train(train_X, train_y, regression, loss, n_epochs=num_epochs, batch_size=batch_size,
                            model_name=model_name, checkpoint_dir=args.checkpoint_dir,
                            context=contextual_info)
            elif task == 'next_activity':
                model.train(train_X, train_y, regression, loss, n_epochs=num_epochs, batch_size=batch_size,
                            model_name=model_name, checkpoint_dir=args.checkpoint_dir,
                            context=contextual_info)

        Running_Time = model.running_time
        print("training is done")
        model.load(checkpoint_dir, model_name=model_name)

        if args.load_saved_test_data:
            with open("%s/%s-%s.pkl" % (test_set_dir, feature_name, "X"), "rb") as text_X_file:
                test_X = pickle.load(text_X_file)
            with open("%s/%s-%s.pkl" % (test_set_dir, feature_name, "y"), "rb") as text_y_file:
                test_y = pickle.load(text_y_file)
            print("flag: test data is loaded")

        else:
            ## loading test data
            filename = test_set_dir + test_data_set + "_%d.csv" % (i+2)
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

            if args.save_data:
                with open("%s/test_%s-%s.pkl" % (test_set_dir, feature_name, "X"), 'wb') as f:
                    pickle.dump(test_X, f)
                with open("%s/test_%s-%s.pkl" % (test_set_dir, feature_name, "y"), 'wb') as f:
                    pickle.dump(test_y, f)
                print("flag: test set is saved")

        exp_info = {"task": task, "filename": filename, "control_flow_p": control_flow_p, "time_p": time_p,
                    "resource_p": resource_p, "data_p": data_p, "transition": transition, "num_epochs": num_epochs,
                    "batch_size": batch_size}

        # Evaluate the model on the test data using `evaluate`
        CF_matrix, report, Accuracy, F1_Score, Precision, Recall = model.evaluate(test_X, test_y, exp_info)
        # with open(results_file, 'w') as f:
        #     f.write("Running time is %.5f \n" % Running_Time)
        #     if not args.load_saved_data:
        #         f.write("Feature Generator time is %.5f \n" % FeatureGenerator_Time)
        #         f.write("Feature Encoder time is %.5f \n" % FeatureEncoder_Time)
        #     else:
        #         FeatureGenerator_Time = 'null'
        #         FeatureEncoder_Time = 'null'
        #     f.write(str(CF_matrix))
        #     f.write("\n")
        #     f.write(str(report))
        #     f.write("\n")
        #     f.write("Accuracy = %.5f \n" % Accuracy)
        #     f.write("F1_Score = %.5f \n" % F1_Score)
        #     f.write("Precision = %.5f \n" % Precision)
        #     f.write("Recall = %.5f \n" % Recall)
        #     f.close()

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

        Results_data = [exp_dataset, (i+2), Accuracy, F1_Score, Precision, Recall, Running_Time, FeatureGenerator_Time,
                        TruePositive, FalsePositive, FalseNegative, TrueNegative]

        with open(total_results_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(Results_data)

    print("done all 5 validation")
