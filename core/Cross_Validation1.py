import pickle
import Cross_config
from feature_encoder import FeatureEncoder
from feature_generator import FeatureGenerator
from model import net
from sklearn.model_selection import train_test_split
import time

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
    test_set_dir = args.test_dir

    cross_number = args.cross_number
    for i in range(cross_number):
        print("Cross number %d: " % i)

        data_dire = args.data_dir
        folder_name = data_dire.split("/")[-2]
        index = folder_name.find("train")
        results_file_name = folder_name[:index + 6] + "%d_" % i + folder_name[index + 6:]
        data_set = results_file_name + ".csv"

        p_data_dire = args.p_data_dir
        p_folder_name = p_data_dire.split("/")[-2]
        index = p_folder_name.find("train")
        p_results_file_name = p_folder_name[:index + 6] + "%d_" % i + p_folder_name[index + 6:]
        p_data_set = p_results_file_name + ".csv"

        if status == 'o':
          results_file = checkpoint_dir + "Results_" + results_file_name + "_1.txt" 
        elif status == 'p':
          results_file = checkpoint_dir + "Results_" + p_results_file_name + "_1.txt"

        if status == 'o':
            filename = data_dire + data_set
            print(filename)
            model_name = '%s-%s_%s_%s_%s_%s_%s_%s_%s_%s' % (
                status, data_set, args.task, control_flow_p, time_p, resource_p, data_p, transition, num_epochs,
                batch_size)
            feature_name = '%s_%s_%s_%s_%s_%s_%s' % (
                data_set, args.task, control_flow_p, time_p, resource_p, data_p, transition)

        elif status == 'p':
            filename = p_data_dire + p_data_set
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
              with open("%s/%s-%s.pkl" % (data_dire, feature_name, "X"), "rb") as text_X_file:
                  train_X = pickle.load(text_X_file)
              with open("%s/%s-%s.pkl" % (data_dire, feature_name, "y"), "rb") as text_y_file:
                  train_y = pickle.load(text_y_file)
          elif status == 'p':
              ## Load Saved Datasets
              with open("%s/%s-%s.pkl" % (p_data_dire, feature_name, "X"), "rb") as text_X_file:
                  train_X = pickle.load(text_X_file)
              with open("%s/%s-%s.pkl" % (p_data_dire, feature_name, "y"), "rb") as text_y_file:
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
                  with open("%s/%s-%s.pkl" % (data_dire, feature_name, "X"), 'wb') as f:
                      pickle.dump(train_X, f)
                  with open("%s/%s-%s.pkl" % (data_dire, feature_name, "y"), 'wb') as f:
                      pickle.dump(train_y, f)
              elif status == 'p':
                  with open("%s/%s-%s.pkl" % (p_data_dire, feature_name, "X"), 'wb') as f:
                      pickle.dump(train_X, f)
                  with open("%s/%s-%s.pkl" % (p_data_dire, feature_name, "y"), 'wb') as f:
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
            test_folder = test_set_dir.split("/")[-2]
            filename = test_set_dir + test_folder + "_%d.csv" % i
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
        with open(results_file, 'w') as f:
            f.write("Running time is %.5f \n" % Running_Time)
            if not args.load_saved_data:
                f.write("Feature Generator time is %.5f \n" % FeatureGenerator_Time)
                f.write("Feature Encoder time is %.5f \n" % FeatureEncoder_Time)
            f.write(str(CF_matrix))
            f.write("\n")
            f.write(str(report))
            f.write("\n")
            f.write("Accuracy = %.5f \n" % Accuracy)
            f.write("F1_Score = %.5f \n" % F1_Score)
            f.write("Precision = %.5f \n" % Precision)
            f.write("Recall = %.5f \n" % Recall)
            f.close()
            
        result_file_names = list()
        FeatureEncoder_Times = list()
        Running_Times = list()
        Accuracys = list()
        F1_Scores = list()
        Precisions = list()
        Recalls = list()
        result_file_names.append(results_file + ".csv")
        Running_Times.append("Running time is %.5f," % Running_Time)
        if not args.load_saved_data:
            FeatureEncoder_Times.append("Feature Encoder time is %.5f " % FeatureEncoder_Time)
        else:
            FeatureEncoder_Times = FeatureEncoder_Times.append('null')
        Accuracys.append("Accuracy = %.5f," % Accuracy)
        F1_Scores.append("F1_Score = %.5f," % F1_Score)
        Precisions.append("Precision = %.5f," % Precision)
        Recalls.append("Recall = %.5f" % Recall)
    import os    
    total_results_file = checkpoint_dir + "Total_Results_" + p_results_file_name + ".csv"
    counter=0
    for n,run, enc, acc,fscore,prec,recall in zip(result_file_names, Running_Times, FeatureEncoder_Times, Accuracys, F1_Scores, Precisions, Recalls):
        print(count)
        count= count+1
        if os.path.exists(total_results_file):
            append_write = 'a' # append if already exists
        else:
            append_write = 'w' # make a new file if not
        
        with open(total_results_file, append_write) as f:
            row = "" + n + "," + run + "," + enc+ "," + acc + "," + fscore+ "," + prec + "," + recall 
            f.write(row + '\n')
            f.close()

    print("done all 5 validation")