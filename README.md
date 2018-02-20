# Generating High-Quality Query Suggestion Candidates for Task-Based Search

This repository provides resources developed within the following paper:

> H. Ding, S. Zhang, D. Garigliotti, and K. Balog. Generating High-Quality Query Suggestion Candidates for Task-Based Search, In ECIR'18, March 2018.

These resources allow to reproduce the results presented in the Task-based Query Suggestion paper.

This repository is structured as follows:

 - `data/`: TSV file used for evaluating the query suggestions. It was obtained by post-processing the test collection (details in the paper).

 - `output/`: all the final TSV run files, containing query suggestions generated by different methods and sources used in the paper.


## Results

Results presented in the paper can be obtained by running the evaluation script, indicating the metrics of interest.

```
$ python eval.py 10  # P@10
$ python eval.py 20  # P@20
```

## Crowdsourcing experiments

We seek to measure the quality of question suggestions for task-based search. Please see details below.

![Experiment Layout](https://github.com/iai-group/ecir2018-neuqs/blob/master/images/exp_layout.png)


## Contact

Should you have any question, please contact Heng Ding at heng.ding@whu.edu.cn
