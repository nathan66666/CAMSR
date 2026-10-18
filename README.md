# CAMSR

Official implementation of **CAMSR: Content-Aware State Space Model for Lightweight Image Super-Resolution**.

## Installation

Use a Python virtual environment with an NVIDIA GPU and a compatible CUDA/PyTorch installation. Install the matching `torch` and `torchvision` builds following the [PyTorch instructions](https://pytorch.org/get-started/locally/), then run:

```bash
python -m pip install packaging ninja
python -m pip install -r requirements.txt --no-build-isolation
python -m pip install -e . --no-deps --no-build-isolation
python -c "import torch; from mamba_ssm.ops.selective_scan_interface import selective_scan_fn; assert torch.cuda.is_available()"
```

The model uses the `mamba-ssm` selective-scan CUDA extension. Production training and inference require CUDA; installing CPU-only PyTorch is insufficient. The dependencies are not version-locked; choose a compatible Python, PyTorch, CUDA and `mamba-ssm` combination.

The Python package remains under `basicsr/` to retain the BasicSR training framework. The network and training wrapper are `CAMSR` and `CAMSRModel`.

## Data preparation

Run commands from the repository root. Prepare paired LR/HR images and edit the dataset paths in both YAML files. The default layout is:

```text
datasets/
├── DF2K/
│   ├── DF2K_HR_sub/
│   ├── X4_sub/
│   └── DF2Ksub_GT.txt
├── Set5/
│   ├── GTmod12/
│   └── LRbicx4/
└── Set14/
    ├── GTmod12/
    └── LRbicx4/
```

Use `io_backend.type: disk`. LR and HR filenames must match (`filename_tmpl: '{}'`), and each HR image must be four times the LR size. Each line of the training metadata file starts with an HR filename, optionally followed by image shape metadata.

## Training

```bash
python -m basicsr.train -opt options/train/train_CAMSR_x4.yml
```

The configuration retains the supplied experiment's network dimensions, L1 loss, optimizer, schedule and augmentation settings: x4 SR, batch size 32 and HR crop size 320. Reduce batch size if necessary for available GPU memory.

Resume an existing run:

```bash
python -m basicsr.train -opt options/train/train_CAMSR_x4.yml --auto_resume
```

Multi-GPU training:

```bash
bash scripts/dist_train.sh 2 options/train/train_CAMSR_x4.yml
```

Checkpoints and training states are saved under `experiments/CAMSR_x4_train/`. View logs with `tensorboard --logdir tb_logger`.

## Testing

Set `path.pretrain_network_g` and the test dataset paths in `options/test/test_CAMSR_x4.yml`, then run:

```bash
python -m basicsr.test -opt options/test/test_CAMSR_x4.yml
```

The default checkpoint is `experiments/CAMSR_x4_train/models/net_g_latest.pth`. Testing loads EMA weights (`params_ema`) when available and falls back to `params`. Results are saved under `results/CAMSR_x4_test/`. PSNR and SSIM use the Y channel with a four-pixel border crop.

## Existing checkpoints

Network parameter names and tensor shapes are preserved, so existing `params` / `params_ema` state dictionaries can be loaded with `strict_load_g: true`. YAML files should now use `network_g.type: CAMSR` and `model_type: CAMSRModel`.

For automatic resume from an older experiment directory, retain its experiment name via `--force_yml name=<existing_experiment_name>`, together with `--auto_resume`. Keep the corresponding model checkpoints and training states in that experiment directory.

## Acknowledgements

The training framework is based on [BasicSR](https://github.com/xinntao/BasicSR). Upstream licenses and attribution are retained in `LICENSE.txt` and `LICENSE/`; the BasicSR citation is in `LICENSE/BasicSR.cff`.

Contact: `xyan_lei@163.com`.
