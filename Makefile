.ONESHELL:
.SECONDEXPANSION:
.SECONDARY:
.PHONY: all models examples tests clean .FORCE
# Dummy target to ensure that prerequisite files are built.
.FORCE:

all: models examples

models: models-v2

examples: examples-v2

tests: test-v2

clean:
	shopt -s globstar nullglob
	rm -rf _temp benchmarks models/**/*.pkl

include make/v2.mk
