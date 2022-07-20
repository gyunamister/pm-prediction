# python train.py --status "o" --task "next_activity" --contextual_info False --num_epochs 10 --learning_rate 0.002 --num_folds 1 --batch_size 256 --data_set "bpi_2013_closed_problems.xes.gz_Train_All_2.csv" --data_dir "../samples/" --checkpoint_dir "./checkpoints/" --control_flow_p True --time_p True --resource_p False --data_p False --transition False

# python train.py --status "p" --task "next_activity" --contextual_info False --num_epochs 10 --learning_rate 0.002 --num_folds 1 --batch_size 256 --data_set "bpi_2013_closed_problems.xes.gz_Train_All_2.csv" --data_dir "../samples/" --p_data_set "bpi_2013_closed_problems.xes.gz_Train_log_2_1.csv" --checkpoint_dir "./checkpoints/" --control_flow_p True --time_p True --resource_p False --data_p False --transition False

# python evaluate.py --status "o" --task "next_activity" --contextual_info False --num_epochs 10 --batch_size 256 --data_set "bpi_2013_closed_problems.xes.gz_Train_All_2.csv" --data_dir "../samples/" --test_data_set "bpi_2013_closed_problems.xes.gz_Test_2.csv" --test_data_dir "../samples/" --checkpoint_dir "./checkpoints/" --control_flow_p True --time_p True --resource_p False --data_p False --transition False

# python evaluate.py --status "p" --task "next_activity" --contextual_info False --num_epochs 10 --batch_size 256  --data_set "bpi_2013_closed_problems.xes.gz_Train_All_2.csv" --p_data_set "bpi_2013_closed_problems.xes.gz_Train_log_2_1.csv" --data_dir "../samples/" --test_data_set "bpi_2013_closed_problems.xes.gz_Test_2.csv" --test_data_dir "../samples/" --checkpoint_dir "./checkpoints/" --control_flow_p True --time_p True --resource_p False --data_p False --transition False

task_array=("next_activity" "next_timestamp")
data_set_array=("bpi_2013_closed_problems.xes.gz" "env_permit.xes.gz")
data_set_num_array=("2" "3" "4" "5" "1")
preprocessing_array=("divide_2" "divide_3" "divide_5" "divide_10" "log_2" "log_3" "log_5" "log_10" "unique_2")
algo_array=("basic_CNN" "basic_LSTM" "basic_LR" "basic_RF" "basic_SVR")
pred_array=("state" "trans")
search="True"
edge_threshold="100"
node_threshold="100"
result="BPIC12_exp_result_1"
horizon_array=(1 )
for task in ${task_array[@]}; do
    for data_set in ${data_set_array[@]}; do
        for data_set_num in ${data_set_num_array[@]}; do
            python train.py --status "o" --task $task --contextual_info False --num_epochs 10 --learning_rate 0.002 --num_folds 1 --batch_size 256 --data_set $data_set"_Train_All_"$data_set_num".csv" --data_dir "../samples/" --checkpoint_dir "./checkpoints/" --control_flow_p True --time_p True --resource_p False --data_p False --transition False
            python evaluate.py --status "o" --task $task --contextual_info False --num_epochs 10 --batch_size 256 --data_set $data_set"_Train_All_"$data_set_num".csv" --data_dir "../samples/" --test_data_set $data_set"_Test_"$data_set_num".csv" --test_data_dir "../samples/" --checkpoint_dir "./checkpoints/" --control_flow_p True --time_p True --resource_p False --data_p False --transition False
            for preprocessing in ${preprocessing_array[@]}; do
                python train.py --status "p" --task $task --contextual_info False --num_epochs 10 --learning_rate 0.002 --num_folds 1 --batch_size 256 --data_set $data_set"_Train_All_"$data_set_num".csv" --data_dir "../samples/" --p_data_set $data_set"_Train_"$preprocessing"_"$data_set_num".csv" --checkpoint_dir "./checkpoints/" --control_flow_p True --time_p True --resource_p False --data_p False --transition False

                python evaluate.py --status "p" --task $task --contextual_info False --num_epochs 10 --batch_size 256  --data_set $data_set"_Train_All_"$data_set_num".csv" --p_data_set $data_set"_Train_"$preprocessing"_"$data_set_num".csv" --data_dir "../samples/" --test_data_set $data_set"_Test_"$data_set_num".csv" --test_data_dir "../samples/" --checkpoint_dir "./checkpoints/" --control_flow_p True --time_p True --resource_p False --data_p False --transition False
            done;
        done;
    done;
done;
			