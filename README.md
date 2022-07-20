# A simple evaluation pipeline of predictive business process monitoring techniques

### Overview

This platform supports the evaluation of novel (instance-level) predictive business process monitoring techniques. It supports

- importing event logs,
- generating basic features (e.g., activity history, service time, resource history),
- plugging in model architectures,
- training models, and
- evaluating models.

### How to

1. Clone repository
   
   - ```git clone ```
2. Install dependencies (Python 3.8.8)
   
   - ```pip install -r requirements.txt```
3. How to:
    1. You can find example scripts at <u>./sample_scripts/</u>
      - Run the script with the following flags
        - *--task*: "next_activity", 
        - *--contextual_info*: "True" if you want to exploit contextual features, "False" otherwise (default: False),
        - *--transition*: "True" if the dataset contains transition information, "False" otherwise (default: False),
        - *--learning_rate*: learning rate of optimization algorithm (default: 0.002),
        - *--num_folds*: number of folds in validation (default: 10),
        - *--batch_size*: training batch size of models (default: 256),
        - *--data_dir*: directory where the dataset is located,
        - *--data_set*: name of the dataset,
        - *--checkpoint*: directory where the models are saved (default: "./checkpoints/"),
        - *--control_flow_p*: "True" if you want to use control-flows as features, "False" otherwise (default: True),
        - *--time_p*: "True" if you want to use time as features, "False" otherwise (default: True),
        - *--resource_p*: "True" if you want to use resource as features, "False" otherwise (default: False),
        - *--data_p*: "True" if you want to use data as features, "False" otherwise (default: False),
        - *--cross_num*: number of cross validations
        - *--result_dir*: directory where the experimental results are saved
      - e.g., 
        - ```python sample_scripts/next_activity_with_neural_network.py --task "next_activity" --contextual_info False --num_epochs 10 --batch_size 256 --data_dir "./samples/" --data_set "bpi_2013_closed_problems.xes.gz_Train_All" --test_data_dir "./samples/" --test_data_set "bpi_2013_closed_problems.xes.gz_Test"  --checkpoint_dir "./checkpoints/" --control_flow_p True --time_p True --resource_p False --data_p False --transition False --cross_number 5 --result_dir "./result/"```
        - ```python sample_scripts/next_activity_with_xgboost.py --task "next_activity" --contextual_info False --data_dir "./samples/" --data_set "bpi_2013_closed_problems.xes.gz_Train_All" --test_data_dir "./samples/" --test_data_set "bpi_2013_closed_problems.xes.gz_Test"  --checkpoint_dir "./checkpoints/" --control_flow_p True --time_p True --resource_p False --data_p False --transition False --cross_number 5 --result_dir "./result/"```

    

### Requirements

- We use the datetime format: "%Y.%m.%d %H:%M".
- We assume the input csv file contains the columns named after the xes elements, e.g., concept:name
- We assume csv format dataset where 'case:concept:name' denotes the caseid, 'concept:name' the activity,'org:resource' the resource, 'time:timestamp' the complete timestamp, 'event_@' the event attribute, and 'case_@' the case attribute.
