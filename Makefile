.PHONY: dryrun run dag clean

CONFIG?=profiles/sge/

dryrun:
	snakemake -p -n --workflow-profile $(CONFIG)

dryrun-until:
	snakemake -p -n --until $(UNTIL) --workflow-profile $(CONFIG) 

run:
	qsub run_pipeline.sh

run-until:
	qsub run_pipeline.sh --until $(UNTIL)

dag:
	snakemake --quiet --dag --workflow-profile $(CONFIG) | dot -Tpng > dag.png

clean:
	rm -rf .snakemake

