# macOS (Apple Silicon) Environment Setup

The following steps recreate the working environment we validated on macOS with an M1 processor. They assume you are using Anaconda or Miniconda. Commands use the `py38-colonsp` environment mentioned in the repository guidelines.

## 1. Create and activate the Python 3.8 environment

```
conda create -n py38-colonsp python=3.8 pip -y
conda activate py38-colonsp
```

## 2. Install the core scientific stack with conda
Install PyTorch (CPU build), NumPy/SciPy, OpenCV, Matplotlib, and their auxiliary libraries via conda so that native dependencies are pulled in automatically.

```
conda install -c pytorch -c conda-forge \
    pytorch=1.13.1 torchvision=0.14.1 torchaudio=0.13.1 cpuonly -y

conda install -c conda-forge \
    numpy=1.23 scipy=1.10 matplotlib=3.7 opencv=4.5 pillow packaging \
    pyparsing python-dateutil kiwisolver importlib-resources tqdm scikit-image -y
```

## 3. Install Apple Silicon TensorFlow
The upstream `tensorflow-cpu` wheel is not published for Apple Silicon. Use the Apple-maintained build and Metal acceleration plugin instead.

```
python -m pip install tensorflow-macos==2.8.0 tensorflow-metal==0.4.0
```

## 4. Install project-specific Python packages with pip
After the conda installations above, use the repository requirements file for the remaining pure-Python utilities.

```
python -m pip install -r requirements_py38_MACOS.txt --no-deps
```

## 5. Verify the environment
Run the following checks to confirm the interpreter and key libraries load correctly.

```
python --version
which python
python -m pip --version
```

```
python - <<'PY'
import torch
import torchvision
import cv2
from torch.backends import mps

print("torch:", torch.__version__)
print("torchvision:", torchvision.__version__)
print("OpenCV:", cv2.__version__)
print("MPS available:", mps.is_available())
PY
```

Finally, run one of the repository scripts to ensure the environment integrates with the project (adjust the dataset paths if required).

```
python export.py export_detector_homoAdapt configs/superpoint_colon_export_test.yaml ds4_specular_camera_mask_th005_k50_topk600_toy
```

If the command completes without missing-module errors, the environment is ready.
