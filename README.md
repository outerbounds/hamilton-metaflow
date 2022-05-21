# Installation
```
git clone https://github.com/outerbounds/hamilton-metaflow.git
cd ./absenteeism
```

This flow depends on Metaflow's integration with Anaconda. You can [install conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html#installation) like: 
```
bash Miniconda3-latest-MacOSX-x86_64.sh
```

## Note about visualizing Hamilton flow
In the `flow.py` step called `featurize_and_split` the code is configured to visualize the hamilton DAG.
By default, we assume you have system graphviz installed. 
If you do not have graphviz or don't want to use it you can set the `graphviz_flag` parameter defined in `flow.py` to `false`. 
If you want to visualize the hamilton DAGs in Metaflow Cards here is a link to the recommended approach to [install graphviz at system level](https://graphviz.org/download/). 
* On MacOS: `brew install graphviz`
* On Unix: `sudo apt-get install graphviz`
* More alternatives are discussed [here](https://stackoverflow.com/questions/35064304/runtimeerror-make-sure-the-graphviz-executables-are-on-your-systems-path-aft).

# Run the flow
Run the flow:
```
python ./flow.py --environment=conda run
```

Note that after configuring AWS credentials you can run any steps on AWS Batch using Metaflows `@batch` decorator. You can configure AWS credentials by following the prompts after:
```
aws configure
```

# Inspecting Results
This flow creates several Metaflow [cards](https://docs.metaflow.org/metaflow/visualizing-results/effortless-task-inspection-with-default-cards). Cards are associated with flow steps. These can be viewed locally in the browser by running 
```
python ./flow.py card view <step name>
```

For example, the `start` step displays class label distribution plots:
```
python ./flow.py card view start
```

You can see Hamilton visualizations by using
```
python ./flow.py card view featurize_and_split
python ./flow.py card view feature_importance_merge
```

You can also look at the scores of each modeling step in a pandas DataFrame like: 
``` python 
from metaflow import Flow
run = Flow('FeatureSelectionAndClassification').latest_run
run.data.results
```

# Paper Reference
* [Application of a neuro fuzzy network in prediction of absenteeism at work](https://ieeexplore.ieee.org/document/6263151)
* [Prediction Of Absenteeism At Work With Multinomial Logistic Regression Model](https://www.researchgate.net/publication/358900589_PREDICTION_OF_ABSENTEEISM_AT_WORK_WITH_MULTINOMIAL_LOGISTIC_REGRESSION_MODEL)
* [Identification of Important Features and Data Mining Classification Techniques in Predicting Employee  Absenteeism at Work](http://ijece.iaescore.com/index.php/IJECE/article/view/25232)

# Dataset reference
* https://archive.ics.uci.edu/ml/datasets/Absenteeism+at+work

