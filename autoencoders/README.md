# Knowledge Distillation (KD) for Anomaly Detection (AD) using autoencoders (AE)

## Snakemake

To set up and run snakemake do
```
pip install snakemake
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

## Workflow

The whole workflow is encoded in the `Snakefile`. Below a detailed description is given
with names of the corresponding Snakemake rules given in brackets.

First prepare train/test/validation datasets (`prepare_data`).

Run hyper parameter tuning for a big teacher (`tune_teacher_kd_ae_l1`). The MSE loss
is minimized, so AD might now be optimal. The hyper parameters are number of layers
and number of filters per layer. Additionally, we can check the results of
hyper parameter tuning with Tensorboard (`run_tensorboard`).
Afterwards, we can check the performance (e.g. ROCs with different signals) of the
tuned big teacher (`check_big_teacher_performance`).

Next we need to create data to train a student model. For the student, input is the
same as input to the teacher, but the target is MSE computed between the teacher's
input and prediction. So we need to prepare data by running it through the teacher
and evaluating the loss (`reformat_ae_l1_data`).

Once the training samples for knowledge distillation are created, we can train the
student with CNN teacher (`kd_ae_l1_train`) or with graph-based teacher (`kd_graph_train`)
and we plot corresponding results (`kd_ae_l1_plot`/`kd_graph_train`), e.g. comparison
of MSE with teacher and student's output.

There are several rules (`co_train_*`) that are used to train the teacher and
student simultaneously. The detailed description can be found in the paper draft
https://www.overleaf.com/project/62aae3bc5f417d057ef58391
