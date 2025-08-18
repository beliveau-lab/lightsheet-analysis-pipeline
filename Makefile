.PHONY: dryrun run dag clean

CONFIG?=config.yaml

dryrun:
	snakemake -n -p --profile profiles/sge --configfile $(CONFIG)

run:
	snakemake -p --profile profiles/sge --configfile $(CONFIG)

dag:
	snakemake --quiet --dag --configfile $(CONFIG) | dot -Tpng > dag.png

clean:
	rm -rf .snakemake

