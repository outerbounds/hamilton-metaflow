# Installation
## Option 1: conda
```
conda env create -f ./absenteeism/conda-env.yml
conda activate ham-flow
```

## Option 2: venv
### Create venv
```
python3 venv -m ham-flow
```

### Activate venv
* On Unix or MacOS, using the bash shell: `source ham-flow/bin/activate`

### Install requirements from PyPi
```
pip3 install -r requirements.txt --no-cache
```

## Note about visualizing Hamilton flow
In the `flow.py` step called `featurize_and_split` the code is configured to visualize the hamilton DAG.
To run this you need to have graphviz on your system PATH. 
A recommended approach is to [install graphviz at system level](https://graphviz.org/download/). 
* On MacOS: `brew install graphviz`
* On Unix: `sudo apt-get install graphviz`
* More alternatives are discussed [here](https://stackoverflow.com/questions/35064304/runtimeerror-make-sure-the-graphviz-executables-are-on-your-systems-path-aft).

# Running the Flow
Note: The flow is currently running the `automl` step on AWS using Metaflow's @batch decorator. If you have not done `metaflow configure aws` you will want to comment this decorator out when running this flow. 

```
python ./absenteeism/flow.py --environment=conda run
```
# Inspecting Results
This flow creates several Metaflow [cards](https://docs.metaflow.org/metaflow/visualizing-results/effortless-task-inspection-with-default-cards). Cards are associated with flow steps. These can be viewed locally in the browser by running 
```
python ./absenteeism/flow.py card view <step name>
```

For example, the `start` step displays class label distribution plots:
```
python ./absenteeism/flow.py card view start
```

You can also look at the scores of each modeling step like: 
``` python 
from metaflow import Flow
flow_data = Flow('FeatureSelectionAndClassification').latest_run.data
print(flow_data.mlr_scores)
print(flow_data.xgb_scores)
print(flow_data.nn_scores)
print(flow_data.tpot_scores)
```

# Paper Reference
* [Application of a neuro fuzzy network in prediction of absenteeism at work](https://ieeexplore.ieee.org/document/6263151)
* [Prediction Of Absenteeism At Work With Multinomial Logistic Regression Model](https://www.researchgate.net/publication/358900589_PREDICTION_OF_ABSENTEEISM_AT_WORK_WITH_MULTINOMIAL_LOGISTIC_REGRESSION_MODEL)
* [Identification of Important Features and Data Mining Classification Techniques in Predicting Employee  Absenteeism at Work](http://ijece.iaescore.com/index.php/IJECE/article/view/25232)

# Dataset reference
* https://archive.ics.uci.edu/ml/datasets/Absenteeism+at+work

