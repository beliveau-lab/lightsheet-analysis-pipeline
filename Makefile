.PHONY: dryrun run dag clean

CONFIG?=config.yaml

dryrun:
	snakemake -p -n --configfile $(CONFIG)

run:
	qsub run_pipeline.sh

dag:
	snakemake --quiet --dag --configfile $(CONFIG) | dot -Tpng > dag.png

clean:
	rm -rf .snakemake

