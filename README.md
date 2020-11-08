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
   - ```pip install requirement.txt```
3. Train and evaluate models
   - Go to <u>./core/</u>
   - ```python train.py``` with following flags
     - *--task*: "next_activity" or "next_timestamp" (default: "next_activity"), 
     - *--contextual_info*: "True" if you want to exploit contextual features, "False" otherwise (default: False),
     - *--learning_rate*: learning rate of optimization algorithm (default: 0.002),
     - *--num_folds*: number of folds in validation (default: 10),
     - *--batch_size*: training batch size of models (default: 256),
     - *--data_set*: name of the dataset,
     - *--data_dir*: directory where the dataset is located,
     - *--checkpoint*: directory where the models are saved (default: "./checkpoints/")
   - e.g., 
     - ```python train.py --task "next_activity" --contextual_info False --num_epochs 50 --learning_rate 0.002 --num_folds 10 --batch_size 256 --data_set "modi_BPI_2012_dropna_filter_act.csv" --data_dir "../sample_data/" --checkpoint "./checkpoints/"```

### Requirements

- We use the datetime format: "%Y.%m.%d %H:%M".
- We assume csv format dataset where ['CASE_ID','Activity','Resource', 'StartTimestamp','CompleteTimestamp'] exist as columns.