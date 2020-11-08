import config
from feature_generator import FeatureGenerator

import keras
import pickle
from datetime import datetime
from model import net

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

	level = args.inter_case_level
	#filename = req['name']

	filename = args.data_dir + args.data_set
	model_name = args.data_set + args.task

	contextual_info = args.contextual_info
	if args.task == 'next_activity':
		loss = 'categorical_crossentropy'
		regression = False
	elif args.task == 'next_timestamp':
		loss = 'mae'
		regression = True

	num_epochs = args.num_epochs
	batch_size = args.batch_size
	num_folds = args.num_folds
	# load data
	print("flag: loading data")
	FG = FeatureGenerator()
	df = FG.create_initial_log(filename)
	print("done")

	#feature generation
	train_df = df
	print("flag: generating feature")
	train_df = FG.order_csv_time(train_df)
	train_df = FG.queue_level(train_df)
	# train_df.to_csv('./training_set_with_features.csv')
	state_list = FG.get_states(train_df)
	train_X, train_Y_Event, train_Y_Time = FG.one_hot_encode_history(train_df, args.checkpoint_dir+args.data_set)
	print("done")
	if contextual_info:
		train_context_X = FG.generate_context_feature(train_df,state_list)
		model = net()
		if regression:
			model.train(train_X, train_context_X, train_Y_Time, regression, loss, n_epochs=num_epochs, batch_size=batch_size, num_folds=num_folds, model_name=model_name, checkpoint_dir=args.checkpoint_dir)
		else:
			model.train(train_X, train_context_X, train_Y_Event, regression, loss, n_epochs=num_epochs, batch_size=batch_size, num_folds=num_folds, model_name=model_name, checkpoint_dir=args.checkpoint_dir)
	else:
		model_name += '_no_context_'
		train_context_X = None
		model = net()
		if regression:
			model.train(train_X, train_context_X, train_Y_Time, regression, loss, n_epochs=num_epochs, batch_size=batch_size, num_folds=num_folds, model_name=model_name, checkpoint_dir=args.checkpoint_dir, context=contextual_info)
		else:
			model.train(train_X, train_context_X, train_Y_Event, regression, loss, n_epochs=num_epochs, batch_size=batch_size, num_folds=num_folds, model_name=model_name, checkpoint_dir=args.checkpoint_dir, context=contextual_info)
