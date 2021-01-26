import eval_config
import pickle
from model import net
from feature_encoder import FeatureEncoder
from feature_generator import FeatureGenerator


if __name__ == '__main__':
  args = eval_config.load()
  
  level = args.inter_case_level
  #filename = req['name']
  status = args.status
  task = args.task
  data_dir = args.data_dir
  data_set = args.data_set
  filename = data_dir + data_set
  contextual_info = args.contextual_info

  num_epochs = args.num_epochs
  batch_size = args.batch_size

  control_flow_p = args.control_flow_p
  time_p = args.time_p
  resource_p = args.resource_p
  data_p = args.data_p

  transition = args.transition

  testset_dir = "../testsets/TestData"

  model_name = '%s_%s_%s_%s_%s_%s_%s' %(args.data_set,args.task,control_flow_p,time_p,resource_p,data_p,transition)

  log_config = {"control_flow_p":control_flow_p, "time_p":time_p, "resource_p":resource_p, "data_p":data_p, "transition":transition}

  # import models
  checkpoint_dir=args.checkpoint_dir
  
  if status == 'o':
    model_name = 'o-%s_%s_%s_%s_%s_%s_%s_%s_%s' %(args.data_set,args.task,control_flow_p,time_p,resource_p,data_p,transition,num_epochs,batch_size)
    print(model_name)
  elif status == 'p':
    model_name = 'p-%s_%s_%s_%s_%s_%s_%s_%s_%s' %(args.data_set,args.task,control_flow_p,time_p,resource_p,data_p,transition,num_epochs,batch_size)
    print(model_name)

  model = net()
  
  model.load(checkpoint_dir,model_name=model_name)

##Mozhgan

#   filename = "../sample_data/Testing/BPIC_2012_100_Testing.csv"

#   print("flag: loading data")
#   fg = FeatureGenerator()
#   df = fg.create_initial_log(filename, log_config)
#   print("done")

#   num_events = len(df)
#   num_cases = len(set(df["id"]))

# # feature generation
#   print("flag: generating features")
#   if task == 'next_activity':
#       loss = 'categorical_crossentropy'
#       regression = False
#       feature_type_list = ["activity_history"]
#       df = fg.add_activity_history(df)
#       df = fg.add_next_activity(df)
#   elif task == 'next_timestamp':
#       loss = 'mae'
#       regression = True
#       feature_type_list = ["activity_history"]
#       df = fg.add_activity_history(df)
#       df = fg.add_query_remaining(df)

#   print("done")

#   # training set generation
#   print("flag: encoding features")
#   feature_name = '%s_%s_%s_%s_%s_%s_%s' % (
#     args.data_set, args.task, control_flow_p, time_p, resource_p, data_p, transition)
#   fe = FeatureEncoder()
#   test_X, test_y = fe.preprocessed_one_hot_encode(df, feature_type_list, task, feature_name)
#   print("done")
  
#   with open("%s/%s-%s.pkl" % (testset_dir, feature_name, "X"), 'wb') as f:
#         pickle.dump(test_X, f)
#   with open("%s/%s-%s.pkl" % (testset_dir, feature_name, "y"), 'wb') as f:
#         pickle.dump(test_y, f)

#   print("flag: test set is saved")
  
##Mozhgan

  # import test data
  # # We use the test data that are already enhanced with features. Thus, ignore the feature generation step.
  # print("flag: loading data")
  # fg = FeatureGenerator()
  # df = fg.create_initial_log(filename,log_config)
  # print("done")
  # feature generation
  # print("flag: generating features")
  # if task == 'next_activity':
  #   loss = 'categorical_crossentropy'
  #   regression = False
  #   feature_type_list = ["activity_history"]
  #   df = fg.add_activity_history(df)
  #   df = fg.add_next_activity(df)
  # elif task == 'next_timestamp':
  #   loss = 'mae'
  #   regression = True
  #   feature_type_list = ["activity_history"]
  #   df = fg.add_activity_history(df)
  #   df = fg.add_query_remaining(df)

  # training set generation
  # print("flag: encoding features")
  # fe = FeatureEncoder()
  # test_X, test_y = fe.one_hot_encode(df,feature_type_list,task)
  # print("done")




  testset_name = "BPIC_2012_100_Testing.csv_next_activity_True_True_False_False_False"
  # #  '%s_%s_%s_%s_%s_%s_%s' %(args.data_set,args.task,control_flow_p,time_p,resource_p,data_p,transition)

  with open("%s/%s-%s.pkl" % (testset_dir,testset_name,"X"), "rb") as text_X_file:
    test_X = pickle.load(text_X_file)
  with open("%s/%s-%s.pkl" % (testset_dir,testset_name,"y"), "rb") as text_y_file:
    test_y = pickle.load(text_y_file)

  print(test_X.shape)
  print(test_y.shape)
  exp_info = {"task":task, "filename":filename, "control_flow_p":control_flow_p, "time_p":time_p, "resource_p":resource_p, "data_p":data_p, "transition":transition, "num_epochs":num_epochs, "batch_size":batch_size}

  # Evaluate the model on the test data using `evaluate`
  # o_results = o_model.evaluate(test_X, test_y, exp_info)
  results = model.evaluate(test_X, test_y, exp_info)
