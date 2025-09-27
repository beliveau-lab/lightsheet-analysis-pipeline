.PHONY: dryrun run dag clean

CONFIG?=config.yaml

dryrun:
	snakemake -p -n --configfile $(CONFIG)

dryrun-until:
	snakemake -p -n --until $(UNTIL) --configfile $(CONFIG) 

run:
	qsub run_pipeline.sh

run-until:
	qsub run_pipeline.sh --until $(UNTIL)

dag:
	snakemake --quiet --dag --configfile $(CONFIG) | dot -Tpng > dag.png

clean:
	rm -rf .snakemake

