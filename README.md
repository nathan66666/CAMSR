# CAMSR

CAMSR: Content-Aware State Space Model for Lightweight Image Super-Resolution。
本精简版以 `options/train/mamba_kmeans_013_4_4_pre_l1_1.yml` 为基准，保留 x4 训练与成对图像测试。

## 环境

需要 NVIDIA GPU、兼容的 CUDA/PyTorch，以及 `mamba-ssm` 的 selective-scan CUDA 扩展。
建议使用独立 Python 虚拟环境，先按 PyTorch 官方说明安装与 CUDA 匹配的 torch 和 torchvision，再安装：

```bash
python -m pip install packaging ninja
python -m pip install -r requirements.txt --no-build-isolation
python -m pip install -e . --no-deps --no-build-isolation
python -c "import torch; from mamba_ssm.ops.selective_scan_interface import selective_scan_fn; assert torch.cuda.is_available(); print('CUDA and selective scan available')"
```

原仓库未锁定版本；请以实际 GPU/CUDA 的兼容组合为准。CPU 不支持本模型的 selective-scan 训练/推理。
不再需要 BasicSR 的 DCN、fused_act 或 upfirdn2d 扩展。

## 数据与训练

在仓库根目录执行。训练 YAML 保留原实验参数及路径，运行前必须修改以下本机路径：

- `datasets.train.dataroot_gt`：DF2K HR 子图。
- `datasets.train.dataroot_lq`：对应的 x4 LR 子图。
- `datasets.train.meta_info_file`：HR 文件名列表；每行首项为图像文件名。
- `datasets.val_1` / `val_2` 的 HR、LR 路径：Set5、Set14。

LR/HR 文件名通过 `filename_tmpl: '{}'` 匹配，HR 尺寸应是 LR 的 4 倍。
原配置使用 batch size 32、HR crop 320；显存不足时自行降低 batch size。

```bash
python -m basicsr.train -opt options/train/mamba_kmeans_013_4_4_pre_l1_1.yml
# 恢复训练（需要已有训练状态）：
python -m basicsr.train -opt options/train/mamba_kmeans_013_4_4_pre_l1_1.yml --auto_resume
# 多卡：
bash scripts/dist_train.sh 2 options/train/mamba_kmeans_013_4_4_pre_l1_1.yml
```

权重、训练状态和日志输出到 `experiments/<实验名称>/`。TensorBoard：
`tensorboard --logdir tb_logger`。

## 测试

`options/test/mamba_kmeans_013_4_x4.yml` 与训练配置使用相同网络结构、x4 倍率和 PSNR/SSIM 指标。
修改测试集 HR/LR 路径和 `path.pretrain_network_g` 为实际文件；默认指向训练生成的 `net_g_latest.pth`，优先读取 EMA 参数 `params_ema`。

```bash
python -m basicsr.test -opt options/test/mamba_kmeans_013_4_x4.yml
```

测试结果写入 `results/<实验名称>/`；PSNR/SSIM 均采用 Y 通道并裁去 4 像素边界。

## 精简范围

保留模型、VimIR/SR 训练封装、Adam/调度器/EMA/检查点、成对数据加载与增强、L1 损失、PSNR/SSIM、日志、分布式入口、安装文件及原许可。
删除其他实验 YAML、其他数据集与损失、NIQE/LPIPS/FID、无关 CUDA 算子、模型转换、旧文档、绘图/分析脚本、CSV/PDF 和重复工具。
模型的聚类统计绘图默认关闭，避免每次前向都写入原机器的 `/data5/CAMSR/cluster_stats.png`；网络计算及权重结构保持不变。

本次云机器未提供 GPU、CUDA、数据集或检查点，因此完整训练、断点恢复和测试指标仍需在具备这些资源的机器验证。

## 致谢与许可

基于 [BasicSR](https://github.com/xinntao/BasicSR)。保留 `LICENSE.txt`、`LICENSE/` 与 `CITATION.cff`。
联系：`xyan_lei@163.com`。
