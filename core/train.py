import pickle

import config
from feature_encoder import FeatureEncoder
from feature_generator import FeatureGenerator
from model import net
from sklearn.model_selection import train_test_split

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
    args = config.load()

    status = args.status

    level = args.inter_case_level
    # filename = req['name']
    task = args.task
    contextual_info = args.contextual_info
    num_epochs = args.num_epochs
    batch_size = args.batch_size
    num_folds = args.num_folds

    control_flow_p = args.control_flow_p
    time_p = args.time_p
    resource_p = args.resource_p
    data_p = args.data_p
    transition = args.transition

    if status == 'o':
        filename = args.data_dir + args.data_set
        model_name = '%s-%s_%s_%s_%s_%s_%s_%s_%s_%s' % (
        status, args.data_set, args.task, control_flow_p, time_p, resource_p, data_p, transition, num_epochs,
        batch_size)

    elif status == 'p':
        filename = args.data_dir + args.p_data_set
        model_name = '%s-%s_%s_%s_%s_%s_%s_%s_%s_%s' % (
        status, args.p_data_set, args.task, control_flow_p, time_p, resource_p, data_p, transition, num_epochs,
        batch_size)

    feature_name = '%s_%s_%s_%s_%s_%s_%s' % (
    args.data_set, args.task, control_flow_p, time_p, resource_p, data_p, transition)

    testset_dir = "../testsets"

    log_config = {"control_flow_p": control_flow_p, "time_p": time_p, "resource_p": resource_p, "data_p": data_p,
                  "transition": transition}

    # load data
    print("flag: loading data")
    fg = FeatureGenerator()
    df = fg.create_initial_log(filename, log_config)
    print("done")

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
    fe = FeatureEncoder()
    if status == 'o':
        train_X, train_y = fe.original_one_hot_encode(df, feature_type_list, task, feature_name)
    elif status == 'p':
        train_X, train_y = fe.preprocessed_one_hot_encode(df, feature_type_list, task, feature_name)
    else:
        print("status not defined!")
    print("done")

    # spliting train and test set
    # print("flag:  training sets")
    # train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.3, random_state=42)
    # print("done")

    # export test data if original data is used
    # if status == 'o':
    #     with open("%s/%s-%s.pkl" % (testset_dir, feature_name, "X"), 'wb') as f:
    #         pickle.dump(test_X, f)
    #     with open("%s/%s-%s.pkl" % (testset_dir, feature_name, "y"), 'wb') as f:
    #         pickle.dump(test_y, f)
    #     print("flag: test set is exported")

    print("flag: training model")
    if contextual_info:
        train_df = fg.queue_level(train_df)
        activity_list = fg.get_activities(train_df)
        train_context_X = fg.generate_context_feature(train_df, activity_list)
        model = net()
        if task == 'next_timestamp':
            model.train(train_X, train_y, regression, loss, n_epochs=num_epochs, batch_size=batch_size,
                        num_folds=num_folds, model_name=model_name, checkpoint_dir=args.checkpoint_dir,
                        X_train_ctx=train_context_X)
        elif task == 'next_activity':
            model.train(train_X, train_y, regression, loss, n_epochs=num_epochs, batch_size=batch_size,
                        num_folds=num_folds, model_name=model_name, checkpoint_dir=args.checkpoint_dir,
                        X_train_ctx=train_context_X)
    else:
        train_context_X = None
        model = net()
        if task == 'next_timestamp':
            model.train(train_X, train_y, regression, loss, n_epochs=num_epochs, batch_size=batch_size,
                        num_folds=num_folds, model_name=model_name, checkpoint_dir=args.checkpoint_dir,
                        context=contextual_info)
        elif task == 'next_activity':
            model.train(train_X, train_y, regression, loss, n_epochs=num_epochs, batch_size=batch_size,
                        num_folds=num_folds, model_name=model_name, checkpoint_dir=args.checkpoint_dir,
                        context=contextual_info)
    print("done")
