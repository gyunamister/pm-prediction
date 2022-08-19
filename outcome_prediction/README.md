## Event log sampling for outcome prediction 

Run ```main_outcome_sampling.py``` with the following flags:

	- *--dataset*: dataset name, 
	- *--raw_log_file*: path to the event log in .cvs format containing outcome labels,
	- *--model*: "xgb" or "lstm",
	- *--status*: "p": learning model with sampling "o": learning model with originl dataset,
	- *--sampling*: unique, divide, or log,
	- *--sampling_param* an intiger value for k in sampling methods,
	- *--sampling_feature* one feature from event log for sampling,
	- *--learning_rate*: learning rate of optimization algorithm ,
	- *--K_fold*: number of folds in validation,
	- *--layers*: number of lstm layers,
	- *--opt*: RMSprop or adam,
	- *--rate*: dropout rate,
	- *--batch_size*
	- *--units*
	- *--num_epochs*
