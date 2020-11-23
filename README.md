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
2. Install dependencies (Python 3.6.12)
   - ```pip install -r requirement.txt```
3. Train and evaluate models
   - Go to <u>./core/</u>
   - ```python train.py``` with following flags
     - *--task*: "next_activity" or "next_timestamp" (default: "next_activity"), 
     - *--contextual_info*: "True" if you want to exploit contextual features, "False" otherwise (default: False),
     - *--transition*: "True" if the dataset contains transition information, "False" otherwise (default: False),
     - *--learning_rate*: learning rate of optimization algorithm (default: 0.002),
     - *--num_folds*: number of folds in validation (default: 10),
     - *--batch_size*: training batch size of models (default: 256),
     - *--data_set*: name of the dataset,
     - *--data_dir*: directory where the dataset is located,
     - *--checkpoint*: directory where the models are saved (default: "./checkpoints/"),
     - *--control_flow_p*: "True" if you want to use control-flows as features, "False" otherwise (default: True),
     - *--time_p*: "True" if you want to use time as features, "False" otherwise (default: True),
     - *--resource_p*: "True" if you want to use resource as features, "False" otherwise (default: False),
     - *--data_p*: "True" if you want to use data as features, "False" otherwise (default: False),
   - e.g., 
     - ```python train.py --task "next_activity" --contextual_info False --num_epochs 10 --learning_rate 0.002 --num_folds 1 --batch_size 256 --data_set "test.csv" --data_dir "../sample_data/" --checkpoint "./checkpoints/" --control_flow_p True --time_p True --resource_p False --data_p False --transition False```
   - The experimental results will be saved in '../result/exp_result.csv'

### Requirements

- We use the datetime format: "%Y.%m.%d %H:%M".
- We assume csv format dataset where 'CASE_ID' denotes the caseid,'Activity' the activity,'Resource' the resource, 'StartTimestamp' the start-transition,'CompleteTimestamp' the complete-transition, 'event_@' the event attribute, and 'case_@' the case attribute.