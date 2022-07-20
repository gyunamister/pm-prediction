import eval_config
import pickle
from model import net
from feature_encoder import FeatureEncoder
from feature_generator import FeatureGenerator


if __name__ == '__main__':

  import logging
  import sys

  root = logging.getLogger()
  root.setLevel(logging.DEBUG)

  handler = logging.StreamHandler(sys.stdout)
  handler.setLevel(logging.DEBUG)
  formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
  handler.setFormatter(formatter)
  root.addHandler(handler)
  
  args = eval_config.load()
  
  level = args.inter_case_level
  #filename = req['name']
  status = args.status
  task = args.task
  data_dir = args.data_dir
  data_set = args.data_set
  p_data_set = args.p_data_set
  filename = data_dir + data_set
  # test_data_set = "../samples/bpi_2013_closed_problems.xes.gz_Test_2.csv"
  test_data_dir = args.test_data_dir
  test_data_set = args.test_data_set
  contextual_info = args.contextual_info

  num_epochs = args.num_epochs
  batch_size = args.batch_size

  control_flow_p = args.control_flow_p
  time_p = args.time_p
  resource_p = args.resource_p
  data_p = args.data_p

  transition = args.transition

  # model_name = '%s_%s_%s_%s_%s_%s_%s' %(args.p_data_set,args.task,control_flow_p,time_p,resource_p,data_p,transition)

  log_config = {"control_flow_p":control_flow_p, "time_p":time_p, "resource_p":resource_p, "data_p":data_p, "transition":transition}

  # import models
  checkpoint_dir=args.checkpoint_dir
  
  if status == 'o':
    model_name = 'o-%s_%s_%s_%s_%s_%s_%s_%s_%s' %(args.data_set,args.task,control_flow_p,time_p,resource_p,data_p,transition,num_epochs,batch_size)
    print(model_name)
  elif status == 'p':
    model_name = 'p-%s_%s_%s_%s_%s_%s_%s_%s_%s' %(args.p_data_set,args.task,control_flow_p,time_p,resource_p,data_p,transition,num_epochs,batch_size)
    print(model_name)

  model = net()
  
  model.load(checkpoint_dir,model_name=model_name)

#Mozhgan

  print("flag: loading data")
  fg = FeatureGenerator()
  df = fg.create_initial_log(test_data_dir + test_data_set, log_config)
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

  print("done")

  # training set generation
  print("flag: encoding features")
  feature_name = '%s_%s_%s_%s_%s_%s_%s' % (
    args.data_set, args.task, control_flow_p, time_p, resource_p, data_p, transition)
  fe = FeatureEncoder()
  test_X, test_y = fe.preprocessed_one_hot_encode(df, feature_type_list, task, feature_name)
  print("done")
  
  with open("%s/%s-%s.pkl" % (test_data_dir, feature_name, "X"), 'wb') as f:
        pickle.dump(test_X, f)
  with open("%s/%s-%s.pkl" % (test_data_dir, feature_name, "y"), 'wb') as f:
        pickle.dump(test_y, f)

  print("flag: test set is saved")
  
##Mozhgan

  # import test data
  # # We use the test data that are already enhanced with features. Thus, ignore the feature generation step.
  # print("flag: loading data")
  # fg = FeatureGenerator()
  # df = fg.create_initial_log(test_data_set,log_config)
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




  testset_name = "BPIC2012_train_4_All.csv_next_activity_True_True_False_False_False"
  # #  '%s_%s_%s_%s_%s_%s_%s' %(args.data_set,args.task,control_flow_p,time_p,resource_p,data_p,transition)

  with open("%s/%s-%s.pkl" % (test_data_dir,feature_name,"X"), "rb") as text_X_file:
    test_X = pickle.load(text_X_file)
  with open("%s/%s-%s.pkl" % (test_data_dir,feature_name,"y"), "rb") as text_y_file:
    test_y = pickle.load(text_y_file)

  print(test_X.shape)
  print(test_y.shape)
  from collections import OrderedDict
  exp_info = OrderedDict()
  exp_info['task'] = task
  exp_info['test_data_set'] = test_data_set
  exp_info['model_name'] = model_name
  exp_info['control_flow_p'] = control_flow_p
  exp_info['time_p'] = time_p
  exp_info['resource_p'] = resource_p
  exp_info['data_p'] = data_p
  exp_info['transition'] = transition
  exp_info['num_epochs'] = num_epochs
  exp_info['batch_size'] = batch_size
  # exp_info = {"task":task, "filename":filename, "control_flow_p":control_flow_p, "time_p":time_p, "resource_p":resource_p, "data_p":data_p, "transition":transition, "num_epochs":num_epochs, "batch_size":batch_size}

  # Evaluate the model on the test data using `evaluate`
  # o_results = o_model.evaluate(test_X, test_y, exp_info)
  results = model.evaluate(test_X, test_y, exp_info)
