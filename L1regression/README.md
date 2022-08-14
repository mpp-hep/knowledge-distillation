# Knowledge Distillation for L1 regression
This part is dedicated to study of knowledge distillation techniques on the problem of L1 regression (MET or HT)

## Requirements 
 -  Execute CERN LCG stack to set up the environment properly, and you might also want to set PATH to your local PATH:
 ```
 source /cvmfs/sft.cern.ch/lcg/views/LCG_102cuda/x86_64-centos7-gcc8-opt/setup.sh ; 
 export PATH=/afs/cern.ch/user/X/X_USER/.local/bin:$PATH
 ```
 - Install small library for DL on graphs from Nadya's master
 ```
 pip install git+https://github.com/chernyavskaya/keras-deep-graph-learning
 ```
 - Install mlphel
 ```
 pip install mplhep --user
 ```
 - Install snakemake
 ```
 pip install snakemake --user
 ```
Now you are all set with the environment and can run the code.

## Snakemake
Snakemake is a workflow management system is a tool to create reproducible and scalable data analyses. 
You just specify several rules (python scripts) you want to run, and the rest is being tracked by snakemake. 
If you are uncertain how to you it at first, just run python scripts to start with.

To run snakemake do
```
snakemake -c1 {rule_name}
```
where `-c1` specifies number of cores provided (one in this case).
It became required to specify it in the latest versions of snakemake,
so to make life easier you can add
`alias snakemake="snakemake -c1"` to your `bash/zsch/whatever` profile
and afterwards simply run `snakemake {rule_name}`.

If you want to run a rule, but Snakemake tells you `Nothing to be done`, use `-f`
to force it. Use `-F` to also force all the upstream rules to be re-run.

Good thing to do before running (only has to be done once) is to create
an `output/` directory that is a symbolic link to your `eos` space (or
wherever you have a lot of available space), to be able to store all the data.
```
ln -s {your-eos-path-to-output} output/  
```

For the rules that have wildcards, you cannot run the rule by doing the usual snakemake rule. Insteaf of the rule, you need to run it with the name of the output where the wildcard (parameter) is specified.
For example : 
```
snakemake plots_selective_true_met
```
will run the rule 'selective_sampling_jets' with wildcard being 'true_met'


## Running the code

### Creating dataset 

Normally, you do not need to change anything in the script 'create_L1regression_data.py'. 
This script prepares the data where objects are smeared (smeared_data, smeared_met, smeared_ht, or true_..). 
There is also 'original_met' that is there if we want to really regress to more process dependent MET, but we do not want that as
this would make our trigger biased. So you can safely ignore it, and only work with smeared and true values.
If when you try to create the dataset, you run out of memory because the RAM on your machine is small, you can use the prepared datasets available at:
```
/eos/project/d/dshep/TOPCLASS/L1jetLepData/L1_met_ht_regression/l1_regression_w_sig_train.h5
/eos/project/d/dshep/TOPCLASS/L1jetLepData/L1_met_ht_regression/l1_regression_w_sig_test.h5
```

All information related to the indexing of the new dataset is stored in utils/data_processing.py 


### Applying additional selections
Additional selections might be needed for your training dataset, the code selective_sampling_jets.py does exactly this. Several options are available : 
 - select events based on process (W, QCD, hChToTauNu, hToTauTau)
 - remove events with given number of jets in the given proportions : --fractions_to_keep
 - selectively subsample dataset based on the variable of interest and the threshold. E.g. with --sampling_var=true_met, and --sampling_threshold=110, 
 the script will produce a dataset with true_met spectrum being uniform from 0 to 110, and then falling like in the original dataset.
 Such procedure might be needed if the spectrum is very unbalanced, and during the trainig only the information from this over represented part of the dataset are being learned. 
Any or all of these options can be applied in one go, but they will be processed in the same order as defined above.
 
Most likely, in the end, we will train regression on QCD only, and evaluate its performance on W as signal and QCD as background. 
 
### Training a regression 

In principle, the code is set up to train either MET regression or HT regression.

- MET regression : use all objects in the event (mu, e/g, jets) with their pt,eta,phi and their PID (partcle id), smeared MET, smeared HT, and regress to MET_true/MET_reco, that will serve as the MET correction for the final MET trigger. Data preparation is done in the class METGraphCreator (utils/data_processing.py)
- HT regression : do the same, but train on jets only with their pt,eta,phi + smeared HT, to regress to HT_true/HT_reco.  Data preparation is done in the class HTGraphCreator (utils/data_processing.py), which additionally keeps only jets, removing other objects.

These Graph creators will produce graph_features, graph_adjacency, and graph_labels. The graph_labels have two components, the first is the regression target, and the second is the true variable of interest (true_MET or true_HT) which is used to condition other metrics for monitoring the performance of the training (see below).
 
You might want to do some preprocessing of your input data before the training. Currently, only log transformation is available, you can can pass comma separate names of the features that you would like to apply a transformation on as --log_features = 'pt,met'. 

For the knowledge distillation we decided to do MET regression, so your variable of interest (--variable) is met, not ht. But again, the code is set up such that you can also run everything for HT regression. Appropriate data handling classes (METGraphCreator or HTGraphCreator) are picked up by the code.


### Teacher model 

The teacher model is a graph convolutional neural network with attention defined in nn/models.py. It is built using layers from the keras_deep_learning library (keras_dgl), that you already installed from Nadya's master repo. If MET regression is used, the PID of the objects has to be embedded. This embedding layer is added automatically to the model, if variable of interest (--variable) is met.

The teacher model is set up as a HyperModel from keras hyper tuner, so the hyperparameters of the model can be optimized 
(this will be done once, and these parameters will be fixed to these optimized values through out all knowledge distillation studies). I am using HyperBand algortihm to do the teacher optimization.

You can specify loss function that you would like to use, currently the following loss functions are supported : mse, mae, mape, msle, huber_delta, quantile_tau, dice_epsilon. You can experiment with those, but in general 'huber_1.0' shows good results. Pay attention, that loss functions are implemented in nn/losses.py. Since we have two components in the true_labels, where we are using only the [0] component as a target, we need to re-implement even keras-available losses such as mse.

Additionally, you can add monitoring of theresholded MSE metrics. Class MseThesholdMetric implements calculation of MSE for data that passes variable>threshold. If you train MET regression, MET>threshold, if you train HT regression, then HT>threshold. E.g. --metric_thresholds=100,150 will monitor MSE(target,prediction) for MET/HT>100 and >150. Keep in mind that if you applied log tranformation on met/ht, you will need to reduce these number to log(100+1)/log(150+1). Having these metrics is nice way to monitor that your regression performance improves in all phase space as the training progresses. This is the reason, we need to have a second component of the graph labels passed to the training. 

If you want to train on full dataset and it does not fit in the memory, use the option --use_generator=1. Currently, the generator is written such that graph features, and adjacency matrix are loaded in the RAM, and then the generator only puts a generator batch on the GPU instead of loading a full dataset on the GPU. The full training file has about 5M events. When building an adjacency matrix for MET training, a matrix of 5Mx18x18 is created, and typically this matrix in float32 fits in the RAM (64Gb, maybe even 32Gb).
However, if you have a machine with small RAM (e.g. 16Gb), you will not be able to create such matrix and in this case a generator will have to be rewritten to create adjecency matrix on the fly which of course will be slower. To start with, you can only take a smaller <ins>random</ins> subset of the data in .h5 files by specifying --max_events.


### Evaluating the performance 

To evaluate the performance, use the script analyze_results_teacher.py. It will produce turn-on curves, resolution plots, distibutions of target and prediction, and MET. It will also produce trigger rates and final ROC plot + .txt file with summary of signal efficiency improvement.  For turn-on curves, you specify --inclusive_thresholds that you want to produce turn-on curves for (e.g. MET : 100,120,150 GeV HT :  200,320,450 GeV).

Make sure that when you evaluate the results, you use the same log-tranformation of the features (if any) as used for the training.

You can specify --signal_names and --bg_names that you want to analyze the results for.




