# Knowledge Distillation for L1 regression
This part is dedicated to study of knowledge distillation techniques on the problem of L1 regression (MET or HT)

## Requirements :
 -  Execute cern lcg stack to set up the environment properly:
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

### Creating dataset : 

Typically you do not need to change anything in the script 'create_L1regression_data.py'. 
This script prepares the data where objects are smeared (smeared_data, smeared_met, smeared_ht, or true_..). 
There is also 'original_met' that is there if we want to really regress to more process dependent MET, but in principle we do not want that.





