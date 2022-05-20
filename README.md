# Knowledge Distillation
The repository is dedicated to study of knowledge distillation techniques for L1 trigger usage

## Autoencoders

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
`ln -s {your-eos-path-to-output} output/  `
