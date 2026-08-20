# HeavyEdge-Classify Model
[![HuggingFace](https://img.shields.io/badge/HuggingFace-Model-orange?logo=huggingface)](https://huggingface.co/heavyedge/classifier-v2)
[![GitHub repository](https://img.shields.io/badge/github-repo-blue?logo=github)](https://github.com/heavyedge/classification-model)

Edge classification models for [HeavyEdge-Classify](https://pypi.org/project/heavyedge-classify/).

Provides:
  - Edge classification models.
  - Benchmark results of different models.
  - Example notebooks.

## Usage

This repository provides scripts to train classification models and notebooks to visualize the model performances.

### Cloning the repository

You need:

- `git`
- Python runtime with `pip`

Run the following commands to clone the repository and install the necessary requirements.

```sh
git clone git@github.com:heavyedge/classification-model.git
cd classification-model
pip install -r requirements.txt
```

### Downloading the dataset (Optional)

Run the following commands to download the profile dataset and labels in the `_data` directory.

```sh
export HUGGINGFACE_TOKEN="..."
export LABELS_V2_GDRIVE="..."
./setup.sh
```

### Acquiring the models

The classification models trained by this project can be acquired by downloading them from the [model repository](https://huggingface.co/heavyedge/classifier-v2).
Alternatively, you can train the models yourself if you have downloaded the dataset.

Either approach creates the trained models in the `models/v*` directories.

#### Direct download

You need:

- [Hugging Face CLI](https://huggingface.co/docs/transformers/en/installation)

Run the following command:

```sh
hf download heavyedge/classifier-v2 --repo-type model --local-dir models/v2
```

#### Training the models

You need:

- `make`

Run the following command:

```sh
make models
```

You can test the trained models by running:

```sh
make test
```

### Using the trained model

Once a model is trained, you can pass it to the `heavyedge` command line to perform inference.

```sh
heavyedge classify-predict <input.h5> models/v2/classifiers/minirocket.sigmoid.pkl -o <output>
```

Refer to the [HeavyEdge-Classify](https://pypi.org/project/heavyedge-classify/) documentation.

### Acquiring the built examples

The benchmark results are visualized as notebooks in the `examples` directory.

The notebook outputs are stripped before being stored in this repository.
To check their outputs, you must acquire the built example notebooks.

You can either download the built notebooks from the [GitHub release](https://github.com/heavyedge/classification-model/releases) artifacts, or build the notebooks yourself if you have acquired the preprocessed data.

#### Building the notebooks

You need:

- `make`

```sh
pip install -r examples/requirements.txt
make examples
```

## Contributing

### Configuring git

Configure the local git filter (run once after cloning):

```sh
nbstripout --install --attributes .gitattributes
git config filter.nbstripout.clean "nbstripout"
git config filter.nbstripout.smudge cat
git config filter.nbstripout.required true
```

### Testing

Setting the `HEAVYEDGE_TEST_MODE` environment variable to `1` downloades and trains on only a small subset of data for testing purposes.

```sh
export HEAVYEDGE_TEST_MODE=1
./setup.sh
make models examples test
```

### Building the container image

The `Dockerfile` is provided to facilitate model distribution without sharing secrets.

After downloading the dataset and training the models, build the image with one of the following targets:

- `infer`
  - Includes the trained models (`models`).
  - Includes essential environment for inference.
- `base` (default)
  - Includes the trained models (`models`).
  - Includes the benchmarks and built examples (`benchmarks`, `examples`).
  - Includes essential environment for inference.
  - Includes non-hidden source files.
- `dev`
  - Includes the dataset (`_data`).
  - Includes the trained models (`models`).
  - Includes the benchmarks and built examples (`benchmarks`, `examples`).
  - Includes essential environment for inference.
  - Includes all source files.

### Versioning policy

This repository follows semantic versioning with [Python version specifiers](https://packaging.python.org/en/latest/specifications/version-specifiers/):

```
N.N.N[{a|b|rc}N][.postN][.devN]
```

The following rules apply to the final release versions:

- Major version is raised when the model API is changed in a backwards-incompatible way.
- Minor version is raised when the models are trained with new dataset.
- Patch version is raised when bugs are fixed.

This repository stores source code for multiple major versions.
When new major version breaks CI/CD pipeline (e.g., having conflicting requirements), new repository can be made.

When a new release is made, trained models are deployed to a repository dedicated to each major release.
For example, when `v2.0.0` is released, `models/v2` is uploaded to `classifier-v2` repository.
This applies to other build outputs, e.g., `examples/v2`.
