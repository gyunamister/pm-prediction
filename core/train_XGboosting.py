import pickle
import time
import config
from feature_encoder import FeatureEncoder
from feature_generator import FeatureGenerator
from xgboost import XGBClassifier
import xgboost as xgb
from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score, confusion_matrix, classification_report

args = config.load()

status = args.status
testset_dir = "../testsets/XGBoost"
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
elif status == 'p':
    filename = args.data_dir + args.p_data_set


model_name = '%s-%s_%s_%s_%s_%s_%s_%s_%s_%s' % (
    status, args.data_set, args.task, control_flow_p, time_p, resource_p, data_p, transition, num_epochs,
    batch_size)

feature_name = '%s_%s_%s_%s_%s_%s_%s' % (
    args.data_set, args.task, control_flow_p, time_p, resource_p, data_p, transition)

log_config = {"control_flow_p": control_flow_p, "time_p": time_p, "resource_p": resource_p, "data_p": data_p,
              "transition": transition}

# load data

if status == 'o':
  trainset_name = '%s_%s_%s_%s_%s_%s_%s' %(args.data_set,args.task,
  control_flow_p,time_p,resource_p,data_p,transition)
elif status == 'p':
    trainset_name = '%s_%s_%s_%s_%s_%s_%s' %(args.p_data_set,args.task,
  control_flow_p,time_p,resource_p,data_p,transition)

# with open("%s/%s-%s.pkl" % (testset_dir, trainset_name, "X"), "rb") as text_X_file:
#     train_X = pickle.load(text_X_file)
# with open("%s/%s-%s.pkl" % (testset_dir, trainset_name, "y"), "rb") as text_y_file:
#     train_y = pickle.load(text_y_file)

print("flag: loading data")
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

df.to_csv('./training_set_with_features.csv')
print("done")

# training set generation
print("flag: encoding train features")
fe = FeatureEncoder()
if status == 'o':
    train_X, train_y = fe.original_one_hot_encode_xgb(df, feature_type_list, task, feature_name)
elif status == 'p':
    train_X, train_y = fe.preprocessed_one_hot_encode_xgb(df, feature_type_list, task, feature_name)
else:
    print("status not defined!")
print("done")
data_name = '%s_%s_%s_%s_%s_%s_%s' % (
    args.p_data_set, args.task, control_flow_p, time_p, resource_p, data_p, transition)
with open("%s/%s-%s.pkl" % (testset_dir, data_name, "X"), 'wb') as f:
      pickle.dump(train_X, f)
with open("%s/%s-%s.pkl" % (testset_dir, data_name, "y"), 'wb') as f:
      pickle.dump(train_y, f)

print("flag: train set is saved")


testset_name = "BPIC_2012_100_Testing.csv_True_True_False_False_False"
# #  '%s_%s_%s_%s_%s_%s_%s' %(args.data_set,args.task,control_flow_p,time_p,resource_p,data_p,transition)
with open("%s/%s-%s.pkl" % (testset_dir, testset_name, "X"), "rb") as text_X_file:
    test_X = pickle.load(text_X_file)
with open("%s/%s-%s.pkl" % (testset_dir, testset_name, "y"), "rb") as text_y_file:
    test_y = pickle.load(text_y_file)

# filename = "../sample_data/Testing/BPIC_2012_100_Testing.csv"


# print("flag: loading data")
# fg = FeatureGenerator()
# df = fg.create_initial_log(filename, log_config)
# print("done")

# num_events = len(df)
# num_cases = len(set(df["id"]))

# # feature generation
# print("flag: generating features")
# if task == 'next_activity':
#     loss = 'categorical_crossentropy'
#     regression = False
#     feature_type_list = ["activity_history"]
#     df = fg.add_activity_history(df)
#     df = fg.add_next_activity(df)
# elif task == 'next_timestamp':
#     loss = 'mae'
#     regression = True
#     feature_type_list = ["activity_history"]
#     df = fg.add_activity_history(df)
#     df = fg.add_query_remaining(df)

# print("done")

# # training set generation
# print("flag: encoding test features")
# feature_name = '%s_%s_%s_%s_%s_%s_%s' % (
#   args.data_set, args.task, control_flow_p, time_p, resource_p, data_p, transition)
# fe = FeatureEncoder()
# test_X, test_y = fe.preprocessed_one_hot_encode_xgb(df, feature_type_list, task, feature_name)
# print("done")
# feature_name = 'BPIC_2012_100_Testing.csv_%s_%s_%s_%s_%s' % (
#   control_flow_p, time_p, resource_p, data_p, transition)
# with open("%s/%s-%s.pkl" % (testset_dir, feature_name, "X"), 'wb') as f:
#       pickle.dump(test_X, f)
# with open("%s/%s-%s.pkl" % (testset_dir, feature_name, "y"), 'wb') as f:
#       pickle.dump(test_y, f)

# print("flag: test set is saved")

## create model
model = XGBClassifier()
print('start training')
Start_time = time.time()

# n_folds = 10
# early_stopping = 10
# params = {'eta': 0.02, 'booster':'gbtree', 'max_depth': 5, 'subsample': 0.7,
#  'objective': 'multi:softmax', 'seed': 99, 'silent': 1, 'eval_metric':'mlogloss',
#   'nthread':4 ,'num_class' : 7 }

# xg_train = xgb.DMatrix(train_X, label=train_y);

# model = xgb.cv(params, xg_train, 10, nfold=n_folds, early_stopping_rounds=early_stopping, verbose_eval=1)


model.fit(train_X, train_y, )
print('tranining time is %.3f S' % (time.time()-Start_time))
print(model)
# make predictions for test data
test_time = time.time()
y_pred = model.predict(test_X)
print('Testing time is %.3f S' % (time.time()-test_time))
predictions = [round(value) for value in y_pred]
# evaluate 
print(confusion_matrix(test_y, predictions))

# Print the precision and recall, among other metrics
print(classification_report(test_y, predictions, digits=3))
print("Accuracy: %.5f" % (accuracy_score(test_y, predictions)))
print("F1-Score: %.5f" % f1_score(test_y, predictions, average="weighted"))
print("precision: %.5f" %precision_score(test_y, predictions, average="weighted"))
print("recall: %.5f" %recall_score(test_y, predictions, average="weighted"))

# print(predictions)
# print("**********")
# print(test_y)
# recall = recall_score(test_y, predictions)
# precision = precision_score(test_y, predictions)
# print("Accuracy: %.2f%% , precision: %.2f%% , recall: %.2f%%" % (accuracy, precision, recall))