.PHONY: models-v2 examples-v2 test-v2

DATASETS_v2 := $(if $(filter 1,$(HEAVYEDGE_TEST_MODE)),dataset5,$(shell ls -d _data/v2/profiles/mean_profiles/dataset* | xargs -n 1 basename))
PROFILES_v2 = $(shell ls _data/v2/profiles/mean_profiles/$(1)/*.h5)
N_SPLITS := $(if $(filter 1,$(HEAVYEDGE_TEST_MODE)),2,5)
TRAIN_JOBS ?= 1
CALIBRATION_METHODS_v2 := sigmoid isotonic sigmoid_ovo isotonic_ovo temperature

models-v2: $(foreach method,$(CALIBRATION_METHODS_v2),models/v2/classifiers/minirocket.$(method).pkl)

examples-v2: $(wildcard examples/v2/*.ipynb)

test-v2: $(foreach method,$(CALIBRATION_METHODS_v2),models/v2/classifiers/minirocket.$(method).pkl)
	@out=$$(mktemp).csv
	trap 'rm -f $$out' EXIT INT TERM
	for model in $^; do
		echo "Testing $$model.."
		heavyedge --log-level=INFO classify-predict _data/v2/profiles/mean_profiles/dataset5/001.h5 $$model -o $$out
	done

_temp/v2/MeanProfiles.h5: $(foreach dataset,$(DATASETS_v2),$(call PROFILES_v2,$(dataset)))
	mkdir -p $(@D)
	heavyedge merge $^ -o $@

_temp/v2/knees.csv: $(foreach dataset, $(DATASETS_v2), _data/v2/labels/$(dataset)/knees.csv)
	mkdir -p $(@D)
	python3 -c "import pandas as pd; dfs = [pd.read_csv(path) for path in '$^'.split()]; pd.concat(dfs)[['Type']].to_csv('$@', index=False)"

_temp/v2/canonical.csv: $(foreach dataset, $(DATASETS_v2), _data/v2/labels/$(dataset)/canonical.csv)
	mkdir -p $(@D)
	python3 -c "import pandas as pd; dfs = [pd.read_csv(path) for path in '$^'.split()]; pd.concat(dfs)[['Type']].to_csv('$@', index=False)"

_temp/v2/labels.csv: scripts/v2/write-labels.py _temp/v2/knees.csv _temp/v2/canonical.csv
	python3 $^ -o $@

models/v2/classifiers/minirocket.%.pkl: _temp/v2/MeanProfiles.h5 _temp/v2/labels.csv
	mkdir -p $(@D)
	heavyedge --log-level=INFO classify-train --n-splits $(N_SPLITS) --calibration $* --n-jobs $(TRAIN_JOBS) --random-state 42 $^ -o $@

_temp/v2/cv-splits.csv: scripts/v2/cv-splits.py _temp/v2/MeanProfiles.h5 _temp/v2/labels.csv
	mkdir -p $(@D)
	python3 $^ --n-splits $(N_SPLITS) -o $@

benchmarks/v2/CV.%.csv: scripts/v2/cv.py _temp/v2/MeanProfiles.h5 _temp/v2/labels.csv _temp/v2/cv-splits.csv
	mkdir -p $(@D)
	python3 $^ --calibration=$* --n-splits $(N_SPLITS) -o $@

benchmarks/v2/CalibrationCurve.%.csv: scripts/v2/calibration-curve.py _temp/v2/labels.csv benchmarks/v2/CV.%.csv
	python3 $^ --n-bins 5 -o $@

benchmarks/v2/CalibrationScores.%.csv: scripts/v2/calibration-scores.py _temp/v2/labels.csv _temp/v2/cv-splits.csv benchmarks/v2/CV.%.csv
	python3 $^ -o $@

examples/v2/profiles.h5: \
_data/v2/profiles/mean_profiles/dataset1/013.h5 \
_data/v2/profiles/mean_profiles/dataset5/013.h5 \
_data/v2/profiles/mean_profiles/dataset5/033.h5 \
_data/v2/profiles/mean_profiles/dataset5/016.h5 \
_data/v2/profiles/mean_profiles/dataset2/017.h5 \
_data/v2/profiles/mean_profiles/dataset2/356.h5 \
_data/v2/profiles/mean_profiles/dataset1/027.h5
	mkdir -p $(@D)
	heavyedge merge $^ -o $@

examples/v2/profiles.ipynb: examples/v2/profiles.h5
	jupyter nbconvert --to notebook --execute --inplace $@

examples/v2/calibration_curve.ipynb: $(foreach method,$(CALIBRATION_METHODS_v2),benchmarks/v2/CalibrationCurve.$(method).csv) .FORCE
	jupyter nbconvert --to notebook --execute --inplace $@

examples/v2/calibration_scores.ipynb: $(foreach method,$(CALIBRATION_METHODS_v2),benchmarks/v2/CalibrationScores.$(method).csv) .FORCE
	jupyter nbconvert --to notebook --execute --inplace $@
