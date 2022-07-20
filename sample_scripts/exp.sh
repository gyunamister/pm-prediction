

python sample_scripts/next_activity_with_neural_network.py --task "next_activity" --contextual_info False --num_epochs 10 --batch_size 256 --data_dir "./samples/" --data_set "bpi_2013_closed_problems.xes.gz_Train_All" --test_data_dir "./samples/" --test_data_set "bpi_2013_closed_problems.xes.gz_Test"  --checkpoint_dir "./checkpoints/" --control_flow_p True --time_p True --resource_p False --data_p False --transition False --cross_number 5 --result_dir "./result/"

python sample_scripts/neural_network.py --task "next_timestamp" --contextual_info False --num_epochs 10 --batch_size 256 --data_dir "./samples/" --data_set "bpi_2013_closed_problems.xes.gz_Train_All" --test_data_dir "./samples/" --test_data_set "bpi_2013_closed_problems.xes.gz_Test"  --checkpoint_dir "./checkpoints/" --control_flow_p True --time_p True --resource_p False --data_p False --transition False --cross_number 5 --result_dir "./result/"

python sample_scripts/next_activity_with_xgboost.py --task "next_activity" --contextual_info False --data_dir "./samples/" --data_set "bpi_2013_closed_problems.xes.gz_Train_All" --test_data_dir "./samples/" --test_data_set "bpi_2013_closed_problems.xes.gz_Test"  --checkpoint_dir "./checkpoints/" --control_flow_p True --time_p True --resource_p False --data_p False --transition False --cross_number 5 --result_dir "./result/"

task_array=("next_activity" "next_timestamp")
# data_set_array=("bpi_2013_closed_problems.xes.gz" "env_permit.xes.gz")
data_set_array=("env_permit.xes.gz" "scenario1_1000_all_0.1.xes.gz" "sudden_trace_noise5_1000_cd.xes.gz")
preprocessing_array=("divide_2" "divide_3" "divide_5" "divide_10" "log_2" "log_3" "log_5" "log_10" "unique_2")

for task in ${task_array[@]}; do
    for data_set in ${data_set_array[@]}; do
            python Cross_Validation.py --status "o" --task $task --contextual_info False --num_epochs 10 --batch_size 256  --data_set $data_set"_Train_All" --p_data_set $data_set"_Train_log_2" --data_dir "../samples/" --test_data_set $data_set"_Test" --test_data_dir "../samples/" --checkpoint_dir "./checkpoints/" --control_flow_p True --time_p True --resource_p False --data_p False --transition False --cross_number 4
            for preprocessing in ${preprocessing_array[@]}; do

                python Cross_Validation.py --status "p" --task $task --contextual_info False --num_epochs 10 --batch_size 256  --data_set $data_set"_Train_All" --p_data_set $data_set"_Train_"$preprocessing --data_dir "../samples/" --test_data_set $data_set"_Test" --test_data_dir "../samples/" --checkpoint_dir "./checkpoints/" --control_flow_p True --time_p True --resource_p False --data_p False --transition False --cross_number 4
        done;
    done;
done;
			