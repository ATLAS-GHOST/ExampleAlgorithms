# ExampleAlgorithms (ATLAS-GHOST)

Example algorithms that could be deployed in **A-GHOST**, inspired by the exisitng ATLAS Global Trigger.

Repository layout (main branch):
- `train_jetfiner.py` — entry-point script for training / running the jet-finder example.
- `train_tower_CNN.py` — entry-point script for training / running the tower CNN example.
- `modules/` — model / algorithm building blocks (layers, networks, etc.).
- `utils/` — utilities (I/O, preprocessing, plotting, metrics, helpers).

> If you are looking for “how do I run this?”, start from **Quickstart** below.

---

## Quick start

Running the code and all the infrastructure depends on having the correct docker file and conda environments. For that a dedicated repository has been created focusing on automating the environments required by the models to be trained and cross checked.

### Environment repository

The dedicated docker containers are distributed via the [CERN harbor registry](https://registry.cern.ch/harbor/projects/3736/repositories). For the models currently hosted here the **train_v2** container is used. The container implements conda environments covering all the tools that are needed in order to execute the code with different flavours of the commercial software that the model relies on

1. base_env: Contains basic tools shared among all environments
2. tf_v2: Contains tools required to generate ML models for FPGAs by utilizing TensorFlow v2 (+Quantizers, +HLS4ML)
3. tf_v3: Contains tools required to generate ML models for FPGAs by utilizing TensorFlow v3 (+Quantizers, +HLS4ML)
4. pytorch: Contains tools required to generate ML models for FPGAs by utlizing pyTorch v2 (+Quantizers)
5. xgboost: Contains tools required to generate BDT models for FPGAs by utilizing XGBoost (+Quantizers, +Conifer)

An overview of how to launch and work with the container can be found in the dedicated repository by the [ATLAS NGT WP2.1 group](https://gitlab.cern.ch/atlas-nextgen-wp21/wp21_ml_framework/-/tree/test_train_v2?ref_type=heads) - branch: **test_train_v2**

**Comment**
There is a method to generate an apptainer wrapper which can simplify execution in different host machines where docker accounts are available.

### Executing the models

All provided models are implemented with the use of TensorFlow and hence are coded using the Model-API. The details of how the models are operating can be found in the modules folder. However the overlay script to train and save the models can be found in the top directory. The instractions below assume that the user has successfully integrated the container environment and is able to launch the docker/apptainer image:

```
#Within the wp21_ml_framework
$ source setup.sh <config>.xml
$ drun/arun
Apptainer> cd /workspace/workDir
Apptainer> tf_v2
Apptainer> python train_jetfinder.py -b 512 -l 2 -w 7 3 -c 16 32 -m 32 16 -e 50

#Results after training are stored automatically in out/ folder
Apptainer> cd out/
```

**To-Be Provided**: Soon a jupyter notebook will be provided showcasing some of the metrics that are important for the models

### Samples

Open data sample are available for this exercise currently hosted via the CERN EOS service. However, access has to be granted to the users by the admins of the group hence get in touch for more details.
