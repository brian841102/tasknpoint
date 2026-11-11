<div align="center">
<h1> TaskNPoint：几分钟教会人形机器人打反手（中文版 · 服务器无头训练指南）</h1>
</div>

<div align="center">
  🌐 <a href="https://ilonadem.github.io/tasknpoint_website/">项目主页</a> |
  📚 <a href="https://arxiv.org/pdf/2606.26215">论文</a>
</div>

> 本文件是英文 [README.md](README.md) 的中文版，并在**没有显示器的 Linux 服务器（无头模式）**上完整实测过一遍：从一段反手击球视频开始，一直到 G1 机器人的训练跑起来。
> 所有步骤都写成**可以直接复制粘贴的命令**，每条命令后面的 `#` 注释说明它的作用。
> 英文 README 里有几条命令**照抄会失败**，本文已经改正，并在 [附录 A：踩坑记录](#附录-a踩坑记录必读) 里逐条说明原因。

---

## 0. 先看这里：整体流程 & 为什么不能直接跑英文 README 的训练命令

英文 README 的"单动作训练"命令是：

```bash
uv run train Mjlab-MultiTarget-Tracking-Flat-Unitree-G1 \
    --registry-name demalenk-california-institute-of-technology-caltech/csv_to_npz/backhand \
    --env.scene.num-envs 4096
```

这条命令**跑不起来**，原因有三个：

1. `uv run train` 调用的其实是依赖库 **mjlab 自带的 `train`**，不是本项目的训练脚本。两个包都注册了同名命令 `train`，mjlab 的覆盖了本项目的。结果 `--registry-name` 被忽略，最后报错 `ValueError: ... racket_contact: []`。
2. 本项目的训练脚本**没有 `--registry-name` 参数**，只能用 `--motion-config xxx.toml` 指定数据来源。
3. `demalenk-.../csv_to_npz/backhand` 是作者的**私有 wandb 数据**，外人没有权限下载（报错 `the provided API key cannot access this resource`）。

所以要自己从视频生成反手动作数据，完整流程如下：

```
反手视频.mp4
  │ 第 1 步  PromptHMR（conda 环境 phmr_pt2.6）   → 人体 3D 姿态 results_*.pkl
  │ 第 2 步  选帧 + 抽取                           → retarget/retarget_inputs/<动作名>.npz
  │ 第 3 步  GMR 重定向（uv 环境）                 → retarget/retarget_outputs/<动作名>.csv
  │ 第 4 步  计算击球相位 + 写 motion_lib.py
  │ 第 5 步  csv_to_npz（上传到你自己的 wandb）    → <你的entity>/csv_to_npz/<动作名>
  │ 第 6 步  写训练配置 TOML
  ▼ 第 7 步  训练（GPU，无头）                     → logs/rsl_rl/g1_tracking/<时间>/model_*.pt
```

本仓库已经放好了一份实测可用的配置：动作名 `forehand_new`（右手正手，见 `motion_lib.py`）和训练配置 `forehand_new_only.toml`。**你只需要按本文把自己的数据生成出来，并把 TOML 里的 entity 改成你自己的。** 反手、正手的流程完全相同。

---

## 1. 需要准备的东西

| 项目 | 说明 |
|---|---|
| Linux 服务器 + NVIDIA GPU | 实测：RTX 4090 ×2，驱动 580，CUDA 13.0。训练只占约 2GB 显存（4096 个环境） |
| 仓库自带的 uv 环境 | 已有 `.venv/`；没有的话按第 3 节安装 |
| miniconda | 位于 `~/miniconda3` |
| wandb 账号 | https://wandb.ai ，免费。用于存储动作数据和训练日志 |
| SMPL-X 账号 | https://smpl-x.is.tue.mpg.de 注册，**要点邮件激活链接** |
| SMPL 账号 | https://smpl.is.tue.mpg.de 注册（与上面是不同的网站，要分别注册） |
| 能登录 Google 的浏览器（在你自己的电脑上） | Google Drive 限流时，要手动下载权重（见第 5.3 节） |
| 一段反手击球视频 | 单人、全身入镜、**固定机位**，几秒到几十秒均可（见第 6.1 节） |

---

## 2. 每次打开新终端都要先执行的命令（最重要！）

下面这几行解决了实测中遇到的几个环境问题（原因见附录 A 的 坑2、坑3、坑4、坑16）。**每开一个新终端，先复制执行一次：**

```bash
cd /home/user/tasknpoint                      # 进入仓库根目录（改成你的实际路径）
unset PYTHONPATH                              # 清掉 ROS 等软件注入的 Python 路径，避免包冲突（坑2）
unset AWS_DEFAULT_REGION                      # 防止 PromptHMR 误以为在 AWS 云上而改错路径（坑4）
export PYTHONNOUSERSITE=1                     # 让 Python 忽略 ~/.local 里的旧包（坑3）
export CUDA_VISIBLE_DEVICES=0                 # 只用 0 号 GPU（多卡服务器上避免抢别人的卡，按需修改）
export MUJOCO_GL=egl                          # MuJoCo 用 EGL 离屏渲染，无显示器也能运行
nvidia-smi                                    # 看各张卡的空闲显存，挑一张空闲的改上面的 CUDA_VISIBLE_DEVICES
```

---

## 3. 安装训练环境（uv 环境 `tasknpoint`）

### 3.1 克隆仓库（已经克隆过的跳过）

```bash
git clone --recurse-submodules https://github.com/wernerb43/tasknpoint   # 连同子模块一起克隆
cd tasknpoint                                                             # 进入仓库
git submodule update --init                                               # 如果之前没带子模块克隆，补拉子模块
```

### 3.2 安装 uv 环境

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh     # 安装 uv（已安装可跳过；装完重新打开终端）
cd /home/user/tasknpoint                             # 回到仓库根目录
uv sync                                              # 按 uv.lock 创建 .venv 并安装训练依赖（mjlab、torch 等）
```

### 3.3 安装 GMR（重定向库）

```bash
uv pip install -e submodules/GMR                     # 把 GMR 装进 uv 环境（uv sync 不会自动装它）
uv run python -c "import general_motion_retargeting, mujoco; print('GMR ok', mujoco.__version__)"   # 验证：应输出 GMR ok 3.8.1
```

> 提示：安装 GMR 时会把 mujoco 升级到 3.14，但之后每次 `uv run` 都会自动按锁文件退回 3.8.1，GMR 在 3.8.1 下也能正常工作。**这属于正常现象，不用处理。**

### 3.4 登录 wandb 并查出你的 entity

```bash
cd /home/user/tasknpoint/tasknpoint_project          # 进入训练子项目
uv run wandb login                                   # 按提示粘贴 API key（在 https://wandb.ai/authorize 获取）
uv run python -c "import wandb; print(wandb.Api().default_entity)"   # 打印你的默认 entity，记下来！
```

> ⚠️ **坑8**：数据会上传到**默认 entity**，它可能是团队名（例如 `brian841102-itri`），而**不是用户名**（`brian841102`）。第 6.9 节的 TOML 里要填这里打印出来的值。

---

## 4. 安装视频处理环境（conda 环境 `phmr_pt2.6`）

```bash
cd /home/user/tasknpoint/submodules/PromptHMR        # 进入 PromptHMR 子模块
source ~/miniconda3/etc/profile.d/conda.sh           # 让当前终端能用 conda 命令
unset PYTHONPATH; export PYTHONNOUSERSITE=1          # 必须！否则会混入 ~/.local 里的旧包（坑3）
bash scripts/install.sh --pt_version=2.6 --world-video=true   # 创建环境并安装 torch2.6/xformers/detectron2/sam2 等，约 10-30 分钟
conda activate phmr_pt2.6                            # 激活新环境
python -c "import torch, detectron2, sam2, lietorch, xformers; print(torch.__version__, torch.cuda.is_available())"   # 验证：应输出 2.6.0+cu126 True
```

> 注意：`install.sh` 没有 `set -e`，中间某一步失败了也会继续往下跑，最后照样打印 "Installation completed"。**一定要用上面最后一条命令验证。**

接着安装 ffmpeg，并把 PromptHMR 和 video_processing 注册为 Python 包：

```bash
conda install -c conda-forge ffmpeg -y                                      # 安装 ffmpeg/ffprobe（统计帧数、解码视频用）
cd /home/user/tasknpoint                                                    # 回到仓库根目录
cp video_processing/prompthmr_pyproject.toml submodules/PromptHMR/pyproject.toml   # PromptHMR 原仓库没有打包文件，复制一份进去
pip install -e submodules/PromptHMR --config-settings editable_mode=compat  # 把 PromptHMR 注册为可导入的包
pip install -e video_processing                                             # 把 video_processing 注册为可导入的包
```

---

## 5. 下载模型权重

下面所有路径都相对于 `submodules/PromptHMR/`。

### 5.1 SMPL / SMPL-X 人体模型（需要账号）

官方脚本 `scripts/fetch_smplx.sh` 有两个问题：没激活 conda 环境时会报 `gdown: 未找到命令`；账号密码错了也不报错，压缩包会被直接删掉（坑6）。所以**改用下面的命令手动下载**，每一步都能看到结果。
**在你自己的终端里执行**（需要输入密码）：

```bash
cd /home/user/tasknpoint/submodules/PromptHMR                         # 进入 PromptHMR
mkdir -p data/body_models/smpl                                         # 建目录

# ---- SMPL-X ----
read -p "SMPL-X 用户名(邮箱): " U; read -s -p "SMPL-X 密码: " P; echo   # 输入账号密码（密码不回显）
wget --post-data "username=$U&password=$P" 'https://download.is.tue.mpg.de/download.php?domain=smplx&sfile=models_smplx_v1_1.zip' -O data/body_models/smplx.zip --no-check-certificate   # 下载 SMPL-X
ls -lh data/body_models/smplx.zip                                      # 应约 830MB；只有几 KB 说明账号密码错误或账号未激活
unzip -q data/body_models/smplx.zip -d data/body_models/smplx          # 解压
mv data/body_models/smplx/models/smplx/* data/body_models/smplx/       # 把模型文件移到 smplx/ 目录下
rm -rf data/body_models/smplx/models data/body_models/smplx.zip        # 删除多余目录和压缩包

# ---- SMPL ----
read -p "SMPL 用户名(邮箱): " U; read -s -p "SMPL 密码: " P; echo      # 输入 SMPL 网站的账号密码
wget --post-data "username=$U&password=$P" 'https://download.is.tue.mpg.de/download.php?domain=smpl&sfile=SMPL_python_v.1.1.0.zip' -O data/body_models/smpl/smpl.zip --no-check-certificate   # 下载 SMPL
unzip -q data/body_models/smpl/smpl.zip -d data/body_models/smpl/smpl  # 解压
mv data/body_models/smpl/smpl/SMPL_python_v.1.1.0/smpl/models/basicmodel_neutral_lbs_10_207_0_v1.1.0.pkl data/body_models/smpl/SMPL_NEUTRAL.pkl   # 中性模型
mv data/body_models/smpl/smpl/SMPL_python_v.1.1.0/smpl/models/basicmodel_f_lbs_10_207_0_v1.1.0.pkl data/body_models/smpl/SMPL_FEMALE.pkl        # 女性模型
mv data/body_models/smpl/smpl/SMPL_python_v.1.1.0/smpl/models/basicmodel_m_lbs_10_207_0_v1.1.0.pkl data/body_models/smpl/SMPL_MALE.pkl          # 男性模型
rm -rf data/body_models/smpl/smpl data/body_models/smpl/smpl.zip       # 清理
ls data/body_models/smplx data/body_models/smpl                        # 应看到 SMPLX_NEUTRAL.npz 等，以及 SMPL_*.pkl
```

> 如果密码里有 `&`、`%`、`+` 等特殊字符，`--post-data` 会解析出错，这种情况下请改用官方脚本 `bash scripts/fetch_smplx.sh`（执行前先 `conda activate phmr_pt2.6`）。

把 SMPL-X 软链接给 GMR 用，避免重复下载：

```bash
cd /home/user/tasknpoint                                                            # 回到仓库根目录
mkdir -p submodules/GMR/assets/body_models                                           # 建 GMR 的模型目录
ln -sfn "$(pwd)/submodules/PromptHMR/data/body_models/smplx" submodules/GMR/assets/body_models/smplx   # 软链接
ls submodules/GMR/assets/body_models/smplx/                                          # 验证能看到 SMPLX_*.npz
```

### 5.2 PromptHMR 权重（先用官方脚本试一次）

```bash
cd /home/user/tasknpoint/submodules/PromptHMR        # 进入 PromptHMR
conda activate phmr_pt2.6                            # 激活环境（脚本要用里面的 gdown）
bash scripts/fetch_data.sh                           # 从 Google Drive 批量下载，很可能部分失败（坑7）
```

### 5.3 检查缺哪些文件，逐个补齐

```bash
cd /home/user/tasknpoint/submodules/PromptHMR        # 进入 PromptHMR
for f in \
  data/pretrain/phmr/checkpoint.ckpt \
  data/pretrain/phmr/config.yaml \
  data/pretrain/phmr_vid/prhmr_release_002.ckpt \
  data/pretrain/phmr_vid/prhmr_release_002.yaml \
  data/pretrain/vitpose-h-coco_25.pth \
  data/pretrain/yolo11x.pt \
  data/pretrain/sam_vit_h_4b8939.pth \
  data/pretrain/droidcalib.pth \
  data/pretrain/camcalib_sa_biased_l2.ckpt \
  data/pretrain/sam2_ckpts/sam2_hiera_tiny.pt \
  data/pretrain/sam2_ckpts/keypoint_rcnn_5ad38f.pkl \
  data/body_models/J_regressor_h36m.npy \
  data/body_models/smpl_mean_params.npz \
  data/body_models/smplx2smpl.pkl \
  data/body_models/smplx2smpl_joints.npy \
  data/body_models/smplx/SMPLX_NEUTRAL.npz \
  data/body_models/smplx/SMPLX_neutral_array_f32_slim.npz ; do
  [ -s "$f" ] && echo "OK   $f" || echo "MISS $f"
done                                                 # 逐个检查 17 个必需文件，MISS 表示缺失
```

对每个显示 **MISS** 的文件，按下表补齐。优先用"服务器命令"：

| 文件 | 服务器命令（官方镜像，不受 Google 限流影响） |
|---|---|
| yolo11x.pt | `wget -O data/pretrain/yolo11x.pt https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11x.pt` |
| sam2_hiera_tiny.pt | `mkdir -p data/pretrain/sam2_ckpts && wget -O data/pretrain/sam2_ckpts/sam2_hiera_tiny.pt https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_tiny.pt` |
| keypoint_rcnn_5ad38f.pkl | `mkdir -p data/pretrain/sam2_ckpts && wget -O data/pretrain/sam2_ckpts/keypoint_rcnn_5ad38f.pkl https://dl.fbaipublicfiles.com/detectron2/COCO-Keypoints/keypoint_rcnn_X_101_32x8d_FPN_3x/139686956/model_final_5ad38f.pkl` |
| sam_vit_h_4b8939.pth | `wget -O data/pretrain/sam_vit_h_4b8939.pth https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth` |
| smpl_mean_params.npz | `cp pipeline/gvhmr/hmr4d/network/hmr2/configs/smpl_mean_params.npz data/body_models/`（仓库里自带一份） |

其余文件**只在 Google Drive 上有**。先在服务器上用 gdown 单独重试（限流是按单个文件计算的，隔一段时间往往就能下载）：

```bash
cd /home/user/tasknpoint/submodules/PromptHMR && conda activate phmr_pt2.6    # 进入目录并激活环境
mkdir -p data/pretrain/phmr data/pretrain/phmr_vid                            # 建目录
gdown 1uQwMCkkqtyQIwuBeWwE83XcOlISKbTHB -O data/pretrain/phmr/checkpoint.ckpt                 # PromptHMR 图像模型（2.3GB）
gdown 1P3EEmBDeRRORhBkUhwPZclZxd7U3Z-yo -O data/pretrain/phmr/config.yaml                     # 其配置
gdown 1ArARc4hMpxSSZc0r6JIpXPFkzR6c8uEB -O data/pretrain/phmr_vid/prhmr_release_002.ckpt      # PromptHMR 视频模型
gdown 1t282bZ_VmTSmB4GnivJRyVhLIsvhyscx -O data/pretrain/phmr_vid/prhmr_release_002.yaml      # 其配置
gdown 1ZprPoNXe_f9a9flr0RhS3XCJBfqhFSeE -O data/pretrain/vitpose-h-coco_25.pth                # ViTPose 2D 关键点模型（2.5GB）
gdown 1gfJohwV63O9cvCo9oHWtQgilEqoBBsg1 -O data/body_models/J_regressor_h36m.npy              # 关节回归矩阵
gdown 1B9LtDH_mMWJZn5P2QnG_3PeQtHSIKOPg -O data/body_models/smplx2smpl.pkl                    # SMPL-X→SMPL 顶点映射（580MB，必需）
gdown 14wB62zX-OMF98xQegYsKEEi84W8gA9D0 -O data/body_models/smplx2smpl_joints.npy             # SMPL-X→SMPL 关节映射（必需）
gdown 1v9Qy7ZXWcTM8_a9K2nSLyyVrJMFYcUOk -O data/body_models/smplx/SMPLX_neutral_array_f32_slim.npz   # 精简版 SMPL-X
```

如果还是报 `Too many users have viewed or downloaded this file recently`，就**在自己电脑上用已登录 Google 的浏览器**打开下面的链接下载。如果浏览器也提示次数过多，先点"添加到我的云端硬盘"，再从自己的云端硬盘里下载：

| 文件名 | 浏览器链接 | 放到服务器的位置（相对 PromptHMR/） |
|---|---|---|
| checkpoint.ckpt | https://drive.google.com/file/d/1uQwMCkkqtyQIwuBeWwE83XcOlISKbTHB/view | data/pretrain/phmr/ |
| prhmr_release_002.ckpt | https://drive.google.com/file/d/1ArARc4hMpxSSZc0r6JIpXPFkzR6c8uEB/view | data/pretrain/phmr_vid/ |
| prhmr_release_002.yaml | https://drive.google.com/file/d/1t282bZ_VmTSmB4GnivJRyVhLIsvhyscx/view | data/pretrain/phmr_vid/ |
| vitpose-h-coco_25.pth | https://drive.google.com/file/d/1ZprPoNXe_f9a9flr0RhS3XCJBfqhFSeE/view | data/pretrain/ |
| J_regressor_h36m.npy | https://drive.google.com/file/d/1gfJohwV63O9cvCo9oHWtQgilEqoBBsg1/view | data/body_models/ |
| smplx2smpl.pkl | https://drive.google.com/file/d/1B9LtDH_mMWJZn5P2QnG_3PeQtHSIKOPg/view | data/body_models/ |
| smplx2smpl_joints.npy | https://drive.google.com/file/d/14wB62zX-OMF98xQegYsKEEi84W8gA9D0/view | data/body_models/ |
| SMPLX_neutral_array_f32_slim.npz | https://drive.google.com/file/d/1v9Qy7ZXWcTM8_a9K2nSLyyVrJMFYcUOk/view | data/body_models/smplx/ |

把下载的文件放进本地一个叫 `tasknpoint_uploads` 的文件夹，然后**在你自己电脑的终端**里执行（`user@172.16.60.162` 换成你的服务器）：

```bash
rsync -avP tasknpoint_uploads/ user@172.16.60.162:~/tasknpoint_uploads/   # 上传；源路径末尾的 / 不能漏，否则会多套一层目录（坑9）
```

再**回到服务器**，把文件分发到对应目录：

```bash
U=~/tasknpoint_uploads; P=/home/user/tasknpoint/submodules/PromptHMR/data     # 设置上传目录和目标目录
mkdir -p $P/pretrain/phmr $P/pretrain/phmr_vid $P/body_models/smplx           # 建目录
mv $U/checkpoint.ckpt                     $P/pretrain/phmr/        2>/dev/null   # 图像模型
mv $U/prhmr_release_002.ckpt $U/prhmr_release_002.yaml $P/pretrain/phmr_vid/ 2>/dev/null   # 视频模型及配置
mv $U/vitpose-h-coco_25.pth               $P/pretrain/             2>/dev/null   # ViTPose
mv $U/J_regressor_h36m.npy $U/smplx2smpl.pkl $U/smplx2smpl_joints.npy $P/body_models/ 2>/dev/null   # 人体模型补充文件
mv $U/SMPLX_neutral_array_f32_slim.npz    $P/body_models/smplx/    2>/dev/null   # 精简 SMPL-X
file $P/pretrain/phmr/checkpoint.ckpt $P/pretrain/vitpose-h-coco_25.pth         # 验证：应显示 "Zip archive data"；显示 HTML 说明下到的是报错网页
```

最后**重新执行第 5.3 节开头的检查命令**，确认 17 个文件全部 OK。

---

## 6. 从视频生成击球动作数据（本节以实测成功的正手 `forehand_new` 为例）

> 正手和反手的流程**完全一样**，只是动作名、截取的帧和目标点不同。下面的命令都以实测视频 `videos/tennis/backhand_new.mp4` 为例。这段视频的文件名虽然叫 backhand，内容其实是**右手正手**的连续空挥练习。
> 第一次实测用的是另一段 iStock 素材，生成了 `backhand_custom`。那段数据的手臂 3D 重建是错的（坑20），**不推荐使用**，保留下来只作为反面例子。

### 6.1 拍摄 / 准备视频

视频要求（第一次实测就是因为不满足这些要求而失败的）：

| 要求 | 原因 |
|---|---|
| 单人、全身入镜（头和脚都拍到）、固定机位、光线好 | PromptHMR 要能稳定地检测和跟踪这个人 |
| **从侧面或斜 45° 拍**，不要正对球员 | 正对镜头时，"手臂伸直"和"手臂朝镜头弯曲"在画面上几乎一样，3D 会重建错（坑20） |
| **正常速度**，不要用慢动作；30fps 或 60fps | 训练出来的挥拍速度取决于视频里的挥拍速度 |
| 每一拍之间最好停下来站定 1 秒 | 片段要从静止的准备姿势开始、到静止姿势结束（见 6.3） |

```bash
cd /home/user/tasknpoint                              # 仓库根目录
mkdir -p videos/tennis                                # 视频放在 videos/<会话名>/ 下；"会话名"会成为输出目录名
mv ~/backhand_new.mp4 videos/tennis/backhand_new.mp4  # 把你的视频移过来（左边改成你的视频路径）
source ~/miniconda3/etc/profile.d/conda.sh && conda activate phmr_pt2.6   # 激活环境（为了用 ffprobe/ffmpeg）
ffprobe -v error -show_entries stream=width,height,r_frame_rate,nb_frames -of default=nw=1 videos/tennis/backhand_new.mp4   # 查看分辨率、帧率、总帧数，记下帧率(r_frame_rate)和总帧数(nb_frames)
```

**分辨率超过 1080p（例如 4K）时，先转成 1080p**。PromptHMR 会把所有帧读进内存，4K 视频很慢也很耗内存（坑21）：

```bash
ffmpeg -v error -y -i videos/tennis/backhand_new.mp4 -vf scale=1920:1080 -c:v libx264 -crf 18 -an videos/tennis/backhand_new_1080p.mp4   # 转成 1080p（竖屏视频改成 scale=1080:1920）
ffprobe -v error -show_entries stream=width,height,r_frame_rate,nb_frames -of csv=p=0 videos/tennis/backhand_new_1080p.mp4   # 确认帧数和帧率没变
```

> 实测视频：3840×2160 → 转成 1920×1080，**30fps**，315 帧。`.mp4` 已被 `.gitignore` 忽略，不会被误提交。

### 6.2 第 1 步：PromptHMR 提取人体 3D 姿态

```bash
cd /home/user/tasknpoint                              # 仓库根目录
unset PYTHONPATH AWS_DEFAULT_REGION; export PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=0   # 必须（坑2、3、4）
nohup setsid bash video_processing/run_prompthmr_single.sh videos/tennis/backhand_new_1080p.mp4 --start-frame 0 --end-frame 314 \
    > video_processing/phmr_backhand_new.log 2>&1 < /dev/null &   # 后台运行（断开 SSH 不会中断），一次处理 0~314 帧（314 = 总帧数-1，按你的视频修改）
tail -f video_processing/phmr_backhand_new.log        # 看进度，出现 "world4d.glb" 表示完成（Ctrl+C 只是退出查看）
ls video_processing/human_pose_outputs/tennis/backhand_new_1080p/   # 应看到 results_0_314.pkl
```

说明：
- 指定 `--start-frame/--end-frame` 后就不会分块，直接得到一个 `results_*.pkl`，**不需要**再执行 `fuse_results.sh`。1000 帧以内建议都这样做。
- 第一次运行会从 HuggingFace 自动下载约 1.9GB 的模型（`timm/vit_large_patch14_clip_224.metaclip_2pt5b` 等）。国内网络下可能**卡住几分钟不动，这是正常的**，耐心等（坑10）。
- 日志里出现 `Error reading from ~/.config/Ultralytics/settings.json` 是无害警告，程序会自动重建配置文件。
- 实测耗时：1080p 315 帧约 5 分钟，432p 629 帧约 8 分钟（RTX 4090）。

### 6.3 第 2 步：挑出一次完整挥拍的起止帧

**(a) 用缩略图看动作**，每 4 帧一张：

```bash
cd /home/user/tasknpoint                              # 仓库根目录
ffmpeg -v error -y -i videos/tennis/backhand_new_1080p.mp4 -vf "select='not(mod(n\,4))',scale=240:-1,drawtext=text='%{eif\:n*4\:d}':x=5:y=5:fontsize=24:fontcolor=yellow:box=1:boxcolor=black,tile=8x5" -vsync 0 frames_%02d.jpg   # 每 4 帧截一张拼成大图，左上角黄色数字 = 原视频帧号
ls frames_*.jpg                                       # 下载到本地查看，或用 VSCode 远程打开
```

**(b) 用重建结果画出持拍手的速度曲线**。这一步最重要：**真正击球的那一下一定是整个动作里最快的**，引拍和收拍都比较慢。只看缩略图很容易把正手和反手看反（坑22）：

```bash
cd /home/user/tasknpoint/submodules/PromptHMR && conda activate phmr_pt2.6   # 进入 PromptHMR 目录，激活环境
PKL=../../video_processing/human_pose_outputs/tennis/backhand_new_1080p/results_0_314.pkl FPS=30 python - <<'EOF'
import os, joblib, numpy as np, torch, smplx
r = joblib.load(os.environ["PKL"]); fps = float(os.environ["FPS"])
p = r["people"][list(r["people"].keys())[0]]; w = p["smplx_world"]; pose = np.asarray(w["pose"]); T = len(pose)
m = smplx.create("data/body_models", model_type="smplx", gender="neutral", use_pca=False, num_betas=10, batch_size=T)
J = m(global_orient=torch.zeros(T, 3), body_pose=torch.tensor(pose[:, 3:66], dtype=torch.float32), betas=torch.tensor(np.asarray(w["shape"]), dtype=torch.float32)).joints.detach().numpy()
rw = J[:, 21] - J[:, 0]                                   # 右手腕相对骨盆（左手持拍改成 20）
v = np.linalg.norm(np.diff(rw, axis=0), axis=1) * fps
def elbow(f):
    ua, fa = J[f, 19] - J[f, 17], J[f, 21] - J[f, 19]; return np.degrees(np.arccos(np.dot(-ua, fa) / np.linalg.norm(ua) / np.linalg.norm(fa)))
print("帧段      手腕速度(m/s)            右肘最大角(180=伸直)  手腕在身体 左(+)/右(-)")
for s in range(0, T - 1, 6):
    seg = v[s:s+6]; k = s + int(seg.argmax())
    print(f"{s:3d}-{s+5:3d}  {seg.max():4.2f} {'#'*int(seg.max()*6):18s}  {max(elbow(f) for f in range(s, min(s+6, T))):4.0f}                 {rw[k,0]:+.2f}")
EOF
```

怎么读这张表（以实测输出为例）：
- `###` 最长的几段就是**击球挥拍**。实测在第 144–161、216–233、276–293 帧，速度 2–3 m/s，每约 2.3 秒出现一次。
- 挥拍前的慢速段是**引拍**，挥拍后的慢速段是**收拍**。
- 看挥拍时手腕从哪一侧到哪一侧（右手持拍）：从右（−）快速挥到左（+）是**正手**，从左（+）快速挥到右（−）是**反手**。
- 右肘在随挥或引拍伸展时，最大角度应该能到 **140° 以上**。如果整段都只有 90–125°，说明 3D 重建把手臂弄弯了，要换视频（坑20）。

**(c) 定起止帧**：顺序应该是 **静止的准备姿势 → 引拍 → 击球 → 随挥/收拍 → 静止姿势**。
- **起点和终点都要选速度最低的帧**，不能从上一拍的收拍中途开始（坑23）；
- 击球点不要太靠近片段的结尾，后面要留出随挥和收拍的时间；
- 片段长度 2~4 秒。

> 实测：连续练习中间没有停顿，速度最低、最平稳的姿势是"球拍收在左肩"，所以截取 **172~240 帧**（69 帧，2.3 秒）：收在左肩 → 右侧引拍 → 快速挥拍（第 216–229 帧）→ 回到左肩收拍。

**(d) 可选：把重建的骨架叠加到原视频上**，确认 3D 手臂和画面对得上：

```bash
cd /home/user/tasknpoint/submodules/PromptHMR        # 在 PromptHMR 目录下
PKL=../../video_processing/human_pose_outputs/tennis/backhand_new_1080p/results_0_314.pkl VIDEO=../../videos/tennis/backhand_new_1080p.mp4 FRAMES="176,196,210,218,224,232,240" python - <<'EOF'
import os, joblib, numpy as np, torch, smplx, cv2
from scipy.spatial.transform import Rotation as R
r = joblib.load(os.environ["PKL"]); p = r["people"][list(r["people"].keys())[0]]; c = p["smplx_cam"]
T = len(np.asarray(c["rotmat"])); aa = R.from_matrix(np.asarray(c["rotmat"])[:, :22].reshape(-1, 3, 3)).as_rotvec().reshape(T, 22, 3)
m = smplx.create("data/body_models", model_type="smplx", gender="neutral", use_pca=False, num_betas=10, batch_size=T)
J = m(global_orient=torch.tensor(aa[:, 0], dtype=torch.float32), body_pose=torch.tensor(aa[:, 1:].reshape(T, 63), dtype=torch.float32),
      betas=torch.tensor(np.asarray(c["shape"]), dtype=torch.float32), transl=torch.tensor(np.asarray(c["trans"]), dtype=torch.float32)).joints.detach().numpy()[:, :22]
cap = cv2.VideoCapture(os.environ["VIDEO"]); frames = []
while True:
    ok, fr = cap.read()
    if not ok: break
    frames.append(fr)
f = float(np.asarray(r["camera"]["img_focal"])); cx, cy = r["camera"]["img_center"]; s = frames[0].shape[1] / (2 * cx)   # PromptHMR 内部缩小过画面，要按比例放大（坑24）
uv = np.stack([J[..., 0] / J[..., 2] * f + cx, J[..., 1] / J[..., 2] * f + cy], -1) * s
bones = [(0,1),(0,2),(0,3),(1,4),(2,5),(3,6),(4,7),(5,8),(6,9),(9,12),(12,15),(9,13),(9,14),(13,16),(14,17),(16,18),(17,19),(18,20),(19,21)]
tiles = []
for fi in [int(x) for x in os.environ["FRAMES"].split(",")]:
    img = frames[fi].copy()
    for a, b in bones: cv2.line(img, tuple(uv[fi, a].astype(int)), tuple(uv[fi, b].astype(int)), (0, 0, 255) if (a, b) in [(17,19),(19,21),(14,17)] else (0, 255, 255), 4)
    cv2.putText(img, str(fi), (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 4); tiles.append(cv2.resize(img, (480, 270)))
cv2.imwrite("overlay.jpg", np.concatenate(tiles, 0)); print("saved overlay.jpg")
EOF
```

下载 `submodules/PromptHMR/overlay.jpg` 查看：**红线（右臂）要和画面里的真实手臂重合**，手臂伸直的地方红线也要是直的。

```bash
rm -f /home/user/tasknpoint/frames_*.jpg /home/user/tasknpoint/submodules/PromptHMR/overlay.jpg   # 看完删掉临时图片
```

### 6.4 抽取片段为重定向输入

```bash
cd /home/user/tasknpoint                              # 仓库根目录
unset PYTHONPATH AWS_DEFAULT_REGION; export PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=0   # 必须
bash video_processing/extract_smpl.sh video_processing/human_pose_outputs/tennis/backhand_new_1080p/results_0_314.pkl 172 241 forehand_new   # 参数：pkl 路径、起始帧、结束帧、输出名
ls -la retarget/retarget_inputs/forehand_new.npz      # 应生成这个文件
```

> ⚠️ **结束帧不包含在内**：`172 241` 截取的是第 172~240 帧，共 69 帧。
> ⚠️ **坑11**：生成的 npz 里帧率被**写死为 30**（`mocap_frame_rate=30`）。如果视频不是 30fps，第 6.8 节的 `--input-fps` 一定要填**视频的真实帧率**，否则动作快慢会不对。

### 6.5 第 3 步：GMR 重定向到 G1 机器人（无头）

```bash
cd /home/user/tasknpoint                              # 仓库根目录
unset PYTHONPATH                                      # 必须
uv run python retarget/construct_motion.py \
    --smplx_file retarget/retarget_inputs/forehand_new.npz \
    --robot unitree_g1 \
    --save_path retarget/retarget_outputs/forehand_new \
    --robot_xml robots/retargeting/g1_27dof.xml \
    --headless --yaw_only_align
# 说明：
#   --headless            无显示器模式，不需要 xvfb
#   --yaw_only_align      【必须加】只把起始帧的水平朝向转正，并让最低的脚踩到地面（坑17）。
#                         不加的话，只要片段不是从直立姿势开始，机器人就会后仰、双脚悬空
ls retarget/retarget_outputs/forehand_new.csv         # 验证输出文件存在
```

> `--yaw_only_align` 是本仓库在实测时新增的参数（修改了 `retarget/construct_motion.py`）。不加它时行为与原版完全一样。运行时会打印类似 `Yaw-only align: initial yaw 65.4 deg, z_shift +0.003 m` 的信息。
> GMR 自带的 `--record_video` 录像视角在机器人左后方，手臂常被身体挡住，**不适合拿来检查动作**（坑25），请用下面检查 2 的录像。

**检查 1：用数值确认脚踩在地上、身体没有后仰**（不需要下载视频）：

```bash
cd /home/user/tasknpoint                              # 仓库根目录
CSV=retarget/retarget_outputs/forehand_new.csv uv run python - <<'EOF'
import os, numpy as np, mujoco
from scipy.spatial.transform import Rotation as R
m = np.loadtxt(os.environ["CSV"], delimiter=",")      # 重定向结果
model = mujoco.MjModel.from_xml_path("robots/training/g1_27dof_training_tennis.xml"); d = mujoco.MjData(model)
lf, rf, tb, pb = model.site("left_foot").id, model.site("right_foot").id, model.body("torso_link").id, model.body("pelvis").id
print("帧   骨盆高  左脚离地  右脚离地  躯干前倾(+)/后仰(-)")
for i, row in enumerate(m):
    d.qpos[:3] = row[:3]; q = row[3:7]; d.qpos[3:7] = [q[3], q[0], q[1], q[2]]; d.qpos[7:] = row[7:]
    mujoco.mj_kinematics(model, d)
    H = R.from_euler("z", R.from_matrix(d.xmat[pb].reshape(3, 3)).as_euler("ZYX")[0]).inv()
    z = H.apply(d.xmat[tb].reshape(3, 3)[:, 2]); fwd = np.degrees(np.arctan2(z[0], z[2]))
    if i % 6 == 0: print(f"{i:3d}  {row[2]:.2f}   {d.site_xpos[lf][2]:.3f}    {d.site_xpos[rf][2]:.3f}    {fwd:+.1f}°")
EOF
```

判断标准：**每一行里至少有一只脚离地小于约 0.07 米**（换脚的瞬间两只脚都可能稍微离地），骨盆高度约 0.7–0.8 米，躯干前倾为正值（击球姿势一般前倾 5–25°）。
（实测：第一次不加 `--yaw_only_align` 时，两只脚离地 0.1–0.39 米，骨盆一路升到 0.94 米；加上后每帧较低那只脚离地都在 0.07 米以内。）

**检查 2：渲染带球拍的正面 + 侧面录像**（左半边是正面，右半边是机器人右侧）：

```bash
cd /home/user/tasknpoint                              # 仓库根目录
CSV=retarget/retarget_outputs/forehand_new.csv OUT=retarget/videos/unitree_g1_forehand_new_front_side.mp4 FPS=30 MUJOCO_GL=egl uv run python - <<'EOF'
import os, numpy as np, mujoco, imageio
from scipy.spatial.transform import Rotation as R
m = np.loadtxt(os.environ["CSV"], delimiter=",")
model = mujoco.MjModel.from_xml_path("robots/training/g1_27dof_training_tennis.xml"); d = mujoco.MjData(model)   # 带球拍的模型
model.vis.global_.offwidth = 608; model.vis.global_.offheight = 608      # 放大离屏画布，否则报 framebuffer 太小（坑26）
model.vis.headlight.ambient[:] = 0.5; model.vis.headlight.diffuse[:] = 0.7
pb = model.body("pelvis").id; r = mujoco.Renderer(model, 608, 608)
def qset(row):
    d.qpos[:3] = row[:3]; q = row[3:7]; d.qpos[3:7] = [q[3], q[0], q[1], q[2]]; d.qpos[7:] = row[7:]; mujoco.mj_kinematics(model, d)
qset(m[0]); yaw0 = np.degrees(R.from_matrix(d.xmat[pb].reshape(3, 3)).as_euler("ZYX")[0])
os.makedirs(os.path.dirname(os.environ["OUT"]), exist_ok=True); w = imageio.get_writer(os.environ["OUT"], fps=float(os.environ["FPS"]))
for row in m:
    qset(row); tiles = []
    for az in (yaw0 + 180, yaw0 + 90):                # 正面、机器人右侧
        cam = mujoco.MjvCamera(); cam.lookat[:] = d.xpos[pb] + [0, 0, 0.2]; cam.distance = 2.0; cam.azimuth = az; cam.elevation = -5
        r.update_scene(d, camera=cam); tiles.append(r.render().copy())
    w.append_data(np.concatenate(tiles, 1))
w.close(); print("saved", os.environ["OUT"])
EOF
```

在**你自己电脑**的终端里下载录像，用播放器打开：

```bash
scp user@172.16.60.162:/home/user/tasknpoint/retarget/videos/unitree_g1_forehand_new_front_side.mp4 ./   # 下载到本地当前目录（user@IP 换成你的服务器）
```

确认三点：①双脚着地；②动作顺序是 **静止 → 引拍 → 快速挥拍 → 收拍 → 静止**，不能一开始就在收拍（坑23）；③手臂在该伸直的地方是伸直的（坑20）。
实测正手录像：第 0–5 帧球拍收在左肩 → 第 10–45 帧右臂伸向右后方引拍 → 第 50–55 帧快速挥过身前 → 第 60–68 帧回到左肩。

### 6.6 第 4 步：计算击球相位（phase）

训练时要告诉模型"球拍在动作的什么时刻（相位 0~1）、到达什么位置"。下面的脚本会打印每一帧球拍的速度和位置：

```bash
cd /home/user/tasknpoint                              # 仓库根目录
CSV=retarget/retarget_outputs/forehand_new.csv FPS=30 uv run python - <<'EOF'
import os, numpy as np, mujoco
FPS = float(os.environ["FPS"])                        # ← 视频的真实帧率
m = np.loadtxt(os.environ["CSV"], delimiter=",")      # 每行 = 根位置3 + 根四元数4(xyzw) + 29个关节角
model = mujoco.MjModel.from_xml_path("robots/training/g1_27dof_training_tennis.xml"); d = mujoco.MjData(model)
rc, pb = model.site("racket_contact").id, model.body("pelvis").id
pos, loc = [], []
for row in m:
    d.qpos[:3] = row[:3]; q = row[3:7]; d.qpos[3:7] = [q[3], q[0], q[1], q[2]]; d.qpos[7:] = row[7:]
    mujoco.mj_kinematics(model, d); pos.append(d.site_xpos[rc].copy())
    loc.append(d.xmat[pb].reshape(3, 3).T @ (d.site_xpos[rc] - d.xpos[pb]))   # 球拍点在机器人身体坐标里
pos, loc = np.array(pos), np.array(loc)
v = np.linalg.norm(np.diff(pos, axis=0), axis=1) * FPS
print("帧   球拍速度(m/s)                    前(+)   左(+)/右(-)")
for i in range(len(v)): print(f"{i:3d}  {v[i]:4.2f} {'#'*int(v[i]*6):30s}  {loc[i,0]:+.2f}   {loc[i,1]:+.2f}")
fast = np.where(v > 0.5 * v.max())[0]                 # 快速挥拍段
hit = int(fast[0] + np.argmax(loc[fast[0]:fast[-1] + 1, 0]))   # 挥拍段中球拍最靠前的一帧 = 击球
print(f"总帧数 {len(m)}，快速挥拍段 {fast[0]}~{fast[-1]}，击球帧 {hit}，相位 phase = {hit/(len(m)-1):.3f}")
EOF
```

怎么读：
- `###` 最长的一段就是**快速挥拍**，引拍和收拍都很慢。
- 击球点取挥拍段里**球拍最靠前（"前"那一列最大）**的那一帧。
- 不要直接用"速度开始变快"的那一帧：那时球拍通常还在身体侧后方，还没到击球点。
- 起点和终点的速度都应该很低（< 1 m/s）。

> 实测输出：`快速挥拍段 47~57，击球帧 52，相位 phase = 0.765`（第 52 帧球拍在身前 0.56 米，最靠前）。
> 仓库里的 `forehand_new` 用的是 0.750（第 51 帧，身前 0.53 米），是逐帧脚本写出来之前手工选的，只差 1 帧。因为目标点是在**同一个相位**用 probe 取的，参考动作和目标依然一致，所以不影响训练。你自己的动作直接用脚本算出的值即可。

### 6.7 第 5 步：在 motion_lib.py 登记动作

打开 `tasknpoint_project/src/tasknpoint_project/motion_sets/motion_lib.py`，仓库里已经加好了实测的 `forehand_new` 条目。**如果你用的是自己的视频**，复制这个条目改个名字（反手可以复制 `backhand_custom`），再按下面三步修改：

1. 把 `probe_points=[("racket_contact", 0.750)]` 里的 0.750 改成第 6.6 节算出的相位；
2. 三个 `target_phase_start/end` 以这个相位为中心：位置和朝向用 ±0.004，速度用 ±0.012；
3. 先保留原来的 `target_pos_mean` 和 `target_orientation_mean` 不动，执行完下一步后，再用打印出来的数值替换。

### 6.8 第 6 步：转换为训练格式并上传 wandb

```bash
cd /home/user/tasknpoint/tasknpoint_project           # 训练子项目
unset PYTHONPATH; export CUDA_VISIBLE_DEVICES=0       # 必须
MUJOCO_GL=egl uv run python src/tasknpoint_project/scripts/csv_to_npz.py \
    --input-file ../retarget/retarget_outputs/forehand_new.csv \
    --output-name forehand_new \
    --input-fps 30 \
    --output-fps 50 \
    --render False \
    --motion-config src/tasknpoint_project/motion_sets/motion_train_configs/tennis_only.toml
# 说明：
#   --output-name  必须和 motion_lib.py 里的条目名完全一致（用于打印 probe 点，也是 wandb 上的数据名）
#   --input-fps    视频的真实帧率（坑11）；实测视频是 30
#   --output-fps   训练用 50Hz，不要改
#   --motion-config 只用来确定带球拍的机器人模型，用 tennis_only.toml 即可
```

输出的最后会打印类似下面的内容，并自动上传到 `<你的entity>/csv_to_npz/forehand_new`（同名再次上传会变成新版本 v1、v2…，训练默认使用最新版）：

```
  racket_contact (site)  phase=0.750  frame=85  t=1.700s
    init  frame : x=0.4815  y=-0.5988  z=0.2849
    ori init    : roll=0.1722  pitch=0.6245  yaw=-0.5094  (rad)
```

把 **init frame** 的数值填回 motion_lib.py 里对应的条目：

```python
target_pos_mean={"x": 0.48, "y": -0.60, "z": 0.28},                   # ← 用 init frame 的 x/y/z
target_orientation_mean={"roll": 0.17, "pitch": 0.62, "yaw": -0.51},  # ← 用 ori init 的 roll/pitch/yaw
```

> motion_lib.py 只在训练时读取，**改完不需要重新执行 csv_to_npz**。
> 合理性检查：右手正手的击球点通常在身前右侧（x≈0.5，y<0），反手在身前左侧（y>0）。作者自己的正手目标是 `x=0.5, y=-0.6`。

### 6.9 第 7 步：写训练配置 TOML

仓库已经提供了 `tasknpoint_project/src/tasknpoint_project/motion_sets/motion_train_configs/forehand_new_only.toml`，**只需要把 entity 改成第 3.4 节打印出来的值**：

```bash
cd /home/user/tasknpoint/tasknpoint_project                                   # 训练子项目
TOML=src/tasknpoint_project/motion_sets/motion_train_configs/forehand_new_only.toml   # 要用的配置文件
ENTITY=$(uv run python -c "import wandb; print(wandb.Api().default_entity)" 2>/dev/null | tail -1)   # 自动获取你的 entity
sed -i "s#^train_prefix = .*#train_prefix = \"$ENTITY/csv_to_npz\"#" $TOML   # 写入 TOML
cat $TOML                                                                    # 确认 train_prefix 和 name 都正确
uv run python -m tasknpoint_project.motion_sets.motion_set $TOML             # 打印将要下载的数据路径，应为 <entity>/csv_to_npz/forehand_new
```

自己的新动作：复制一份 TOML，把 `[[motions]]` 下面的 `name` 改成你在 motion_lib.py 里登记的名字。

---

## 7. 训练（无头，后台运行，断开 SSH 也不会停）

```bash
cd /home/user/tasknpoint/tasknpoint_project           # 训练子项目
unset PYTHONPATH                                      # 必须
export CUDA_VISIBLE_DEVICES=0 MUJOCO_GL=egl           # 选卡 + 无头渲染
export PYTHONUNBUFFERED=1                             # 日志实时写入文件（不加的话 Python 会缓冲，日志文件长时间是空的）
mkdir -p logs                                         # 日志目录
nohup setsid uv run python -m tasknpoint_project.scripts.train Mjlab-MultiTarget-Tracking-Flat-Unitree-G1 \
    --motion-config src/tasknpoint_project/motion_sets/motion_train_configs/forehand_new_only.toml \
    --env.scene.num-envs 4096 \
    > logs/train_forehand_new.log 2>&1 < /dev/null &      # 后台启动；setsid+nohup 保证关闭终端后训练继续
sleep 10                                              # 等 Python 进程起来
pgrep -f "[p]ython3 -m tasknpoint_project.scripts.train" > logs/train_forehand_new.pid   # 把训练进程的 PID 存到文件（[p] 写法避免匹配到 pgrep 自己）
cat logs/train_forehand_new.pid                           # 显示 PID
```

> ⚠️ **必须用 `uv run python -m tasknpoint_project.scripts.train`**，不能用 `uv run train`（坑5）。

查看训练状态：

```bash
cd /home/user/tasknpoint/tasknpoint_project                                       # 训练子项目
tail -f logs/train_forehand_new.log | grep --line-buffered -E "Learning iteration|Mean reward|Mean episode length|ETA"   # 实时看进度（Ctrl+C 退出查看，不影响训练）
grep -E "View run at" logs/train_forehand_new.log                                     # 打印 wandb 网页链接，浏览器打开可以看曲线
ls logs/rsl_rl/g1_tracking/                                                       # 每次训练一个时间戳目录
ls logs/rsl_rl/g1_tracking/*/model_*.pt                                           # 模型存档，每 500 轮保存一次
nvidia-smi                                                                        # 看 GPU 占用
echo "日志 $(( $(date +%s) - $(stat -c %Y logs/train_forehand_new.log) )) 秒前更新"     # 正常训练时这个数字只有几秒
```

停止训练：

```bash
cd /home/user/tasknpoint/tasknpoint_project           # 训练子项目
kill $(cat logs/train_forehand_new.pid)               # 按 PID 结束训练进程（已保存的 model_*.pt 不会丢）
pgrep -af "[t]asknpoint_project.scripts.train" || echo "已停止"   # 确认已经停止
```

> ⚠️ 不要用 `pkill -f "tasknpoint_project.scripts.train"`：如果这条命令是放在某个脚本或 `bash -c` 里执行的，它会把**执行它的 shell 自己**也匹配上一起杀掉（坑18）。

**训练卡住的处理**：实测中遇到过一次，训练跑到第 47 轮后进程不再输出，CPU 和 GPU 占用都为 0，原因没能查明（没有 sudo 权限，无法抓调用栈），按上面的命令停止后重新启动就恢复正常了。
如果"日志 N 秒前更新"里的 N 超过 120，并且 `nvidia-smi` 里这个 PID 的 GPU 占用为 0，就停止训练，然后重新执行本节开头的启动命令。

实测数据（RTX 4090，4096 个环境）：显存约 **2GB**；GPU 空闲时每轮约 0.9 秒，**30000 轮约 7.5 小时**；和别人的任务共用一张卡时每轮约 1.6 秒，约 13 小时。
训练正常的表现：第 0 轮 `Mean reward` 为负数（约 −0.7 到 −1.6），几十轮后变成正数，之后持续上升。

如果只想快速验证流程能跑通，可以减少轮数：

```bash
uv run python -m tasknpoint_project.scripts.train Mjlab-MultiTarget-Tracking-Flat-Unitree-G1 \
    --motion-config src/tasknpoint_project/motion_sets/motion_train_configs/forehand_new_only.toml \
    --env.scene.num-envs 4096 --agent.max-iterations 1000     # 只训 1000 轮（约 27 分钟）
```

---

## 8. 其他动作集的训练（原 README 翻译，已改正命令）

所有命令都在 `tasknpoint_project/` 目录下执行。**这些 TOML 默认指向作者的私有 wandb，外人无法直接使用**，需要按第 6 节自己生成数据，并把 TOML 的 `train_prefix` 改成你的 entity。

```bash
cd /home/user/tasknpoint/tasknpoint_project                                     # 训练子项目
ls src/tasknpoint_project/motion_sets/motion_train_configs/                     # 仓库里实际存在的配置：forehand_new_only / backhand_only / kicks_new / kicks_only / pickup_new / tennis_mle / tennis_only / tennis_only_fast
uv run python -m tasknpoint_project.motion_sets.motion_set src/tasknpoint_project/motion_sets/motion_train_configs/tennis_only.toml --list   # 列出配置里启用的动作（原 README 的 uv run motion-set 命令不存在）

# 网球（正手/反手及步法变体，共 6 个动作）：
uv run python -m tasknpoint_project.scripts.train Mjlab-MultiTarget-Tracking-Flat-Unitree-G1 \
    --motion-config src/tasknpoint_project/motion_sets/motion_train_configs/tennis_only.toml \
    --env.scene.num-envs 4096

# 只练踢球：
uv run python -m tasknpoint_project.scripts.train Mjlab-MultiTarget-Tracking-Flat-Unitree-G1 \
    --motion-config src/tasknpoint_project/motion_sets/motion_train_configs/kicks_only.toml \
    --env.scene.num-envs 4096
```

> 原 README 里提到的 `all_motions.toml`、`tennis_only_expanded.toml`、`box_grab.toml` **在仓库中不存在**。

---

## 9. 评估与回放（原 README 翻译 + 服务器用法）

回放同样要用 `python -m`（坑5），不能用 `uv run play`。服务器没有显示器时，程序会自动使用 **viser 网页查看器**。
直接用本地的模型文件（不用从 wandb 下载），后台启动：

```bash
cd /home/user/tasknpoint/tasknpoint_project           # 训练子项目
unset PYTHONPATH; export CUDA_VISIBLE_DEVICES=0 MUJOCO_GL=egl PYTHONUNBUFFERED=1   # 必须
ss -ltn | grep ":8080 " || echo "8080 空闲"            # 确认 8080 端口没被别人占用
RUN=$(ls -d logs/rsl_rl/g1_tracking/20*/ | tail -1); ls $RUN | grep model_ | sort -V | tail -1   # 最新一次训练的目录，以及其中最新的模型
nohup setsid uv run python -m tasknpoint_project.scripts.play Mjlab-MultiTarget-Tracking-Flat-Unitree-G1 \
    --checkpoint-file $RUN/$(ls $RUN | grep model_ | sort -V | tail -1) \
    --motion-config src/tasknpoint_project/motion_sets/motion_train_configs/forehand_new_only.toml \
    --viewer viser > logs/play.log 2>&1 < /dev/null &    # 后台启动回放
sleep 60; grep -E "listening|Traceback" logs/play.log  # 出现 "viser (listening *:8080)" 表示启动成功
pgrep -f "[p]ython3 -m tasknpoint_project.scripts.play" > logs/play.pid   # 记下 PID，停止时要用
# 也可以改用 wandb 上的模型：把 --checkpoint-file ... 换成 --wandb-run-path <你的entity>/mjlab/<run-id>
#   <run-id> 就是训练日志里 "View run at https://wandb.ai/<entity>/mjlab/runs/<run-id>" 最后那一段
```

在**你自己电脑**的浏览器里打开，两种方法选一种：

```bash
# 方法 1：同一个内网时，浏览器直接打开 http://172.16.60.162:8080 （viser 监听的是 0.0.0.0）
# 方法 2：方法 1 打不开时，在自己电脑的终端里做端口转发（窗口保持打开），再打开 http://localhost:8080
ssh -N -L 8080:localhost:8080 user@172.16.60.162      # 把服务器的 8080 端口转发到本机
```

网页显示 **"WebGL is unavailable"** 时（坑27），这是你**本地浏览器**的问题，和服务器无关。在 Chrome 里依次尝试：
1. 打开 `chrome://settings/system`，开启"使用图形加速功能"，然后重启浏览器；
2. 打开 `chrome://flags/#ignore-gpu-blocklist`，设为 Enabled，然后重启浏览器；
3. 还不行（远程桌面、虚拟机、没有显卡），就完全关闭 Chrome，再用命令行启动：`chrome --ignore-gpu-blocklist --enable-unsafe-swiftshader`；
4. 打开 `chrome://gpu`，"WebGL" 那一行显示 Hardware accelerated 或 Software 就可以用了。

看完后停止回放（不停的话会一直占着 8080 端口和约 2GB 显存）：

```bash
cd /home/user/tasknpoint/tasknpoint_project && kill $(cat logs/play.pid)   # 按 PID 停止回放
```

**不用浏览器，直接在服务器上录一段 mp4**（浏览器 WebGL 用不了时推荐）。viser 网页自带的录制功能也要靠浏览器的 WebGL 渲染，所以同样用不了。`--viewer none` 是本仓库新增的选项：录完指定步数就退出，不打开任何查看器。

```bash
cd /home/user/tasknpoint/tasknpoint_project           # 训练子项目
unset PYTHONPATH; export CUDA_VISIBLE_DEVICES=0 MUJOCO_GL=egl   # 必须
RUN=$(ls -d logs/rsl_rl/g1_tracking/20*/ | tail -1)   # 最新一次训练的目录
uv run python -m tasknpoint_project.scripts.play Mjlab-MultiTarget-Tracking-Flat-Unitree-G1 \
    --checkpoint-file ${RUN}$(ls $RUN | grep model_ | sort -V | tail -1) \
    --motion-config src/tasknpoint_project/motion_sets/motion_train_configs/forehand_new_only.toml \
    --num-envs 1 --video True --video-length 300 --video-height 480 --video-width 640 --viewer none
# 说明：
#   --video True          布尔参数必须写 True，只写 --video 会报 invalid choice（坑28）
#   --video-length 300    录 300 步；控制频率 50Hz，也就是 6 秒
#   --video-height/width  480p；不要超过 640×480，超过会报 framebuffer 太小（坑26）
ls ${RUN}videos/play/                                 # 输出 rl-video-step-0.mp4
```

在**你自己电脑**上下载观看：

```bash
scp user@172.16.60.162:/home/user/tasknpoint/tasknpoint_project/logs/rsl_rl/g1_tracking/<训练时间目录>/videos/play/rl-video-step-0.mp4 ./   # <训练时间目录> 换成上面 RUN 的目录名
```

录像里的**绿色球就是目标点**，击球时球拍应该经过它。

### 9.1 评估：量化"多远的球能打到"

评估脚本会在骨盆坐标系的一个立方体范围内随机放 4096 个目标点，每个回合只放一个，统计球拍离目标最近的距离，以及在击球时刻附近（±0.1 秒内）是否打中（误差 ≤ 20 厘米）。
`--checkpoint-file`、`--first-motion-only`、`--per-env-file` 三个参数都是本仓库新增的：

```bash
cd /home/user/tasknpoint/tasknpoint_project           # 训练子项目
nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv   # 挑一张空闲显存 > 3GB 的卡（坑29）
unset PYTHONPATH; export CUDA_VISIBLE_DEVICES=1 MUJOCO_GL=egl   # 按上面结果选卡
mkdir -p logs/eval                                    # 评估结果目录
uv run python -m tasknpoint_project.goal_cond_tracking.scripts.evaluate Mjlab-MultiTarget-Tracking-Flat-Unitree-G1 \
    --checkpoint-file logs/rsl_rl/g1_tracking/2026-09-24_16-58-21/model_29999.pt \
    --motion-config src/tasknpoint_project/motion_sets/motion_train_configs/forehand_new_only.toml \
    --target-x-min 0.28 --target-x-max 0.68 --target-y-min -1.40 --target-y-max 0.20 --target-z-min -0.52 --target-z-max 1.08 \
    --first-motion-only True \
    --output-file logs/eval/forehand_new_2sigma.json --per-env-file logs/eval/forehand_new_2sigma.npz
# 说明：
#   --checkpoint-file     用本地模型（原脚本只能从 wandb 下载，run 被删掉后就用不了了，坑30）
#   --target-*-min/max    目标点范围，相对骨盆，单位米。这里取训练分布 均值±2σ：
#                         x: 0.48±0.2   y: -0.60±0.8   z: 0.28±0.8（均值、σ 见 motion_lib.py 的 target_pos_mean/std）
#   --first-motion-only   【必须加】只统计第一拍。原脚本统计整个 10 秒回合，第一拍打完后目标点会按训练分布
#                         重新随机，结果被后面的简单目标拉高（实测：不加时显示命中 99.9%，加上后为 83.3%，坑31）
#   --per-env-file        保存每个目标点的结果，用于下面的分组统计
```

输出里最重要的是 `combined_success_rate`（在击球时刻附近打中的比例）和 `min_goal_pos_error`（平均最近距离，单位米）。

**按距离和方向分组统计**，找出打不到的球集中在哪里：

```bash
cd /home/user/tasknpoint/tasknpoint_project           # 训练子项目
NPZ=logs/eval/forehand_new_2sigma.npz uv run python - <<'EOF'
import os, numpy as np
r = np.load(os.environ["NPZ"]); g, e, hit = r["goal_pts"], r["min_goal_pos_error"], r["combined_success"]
nom = r["nominal_target"][0]; d = g - nom; dist = np.linalg.norm(d, axis=1)
print(f"标准击球点(骨盆坐标): x={nom[0]:.2f} y={nom[1]:.2f} z={nom[2]:.2f}\n离标准点距离   样本数  命中率  平均最近误差")
for lo, hi in [(0, .2), (.2, .4), (.4, .6), (.6, .8), (.8, 2)]:
    m = (dist >= lo) & (dist < hi)
    if m.sum(): print(f"  {lo:.1f}–{hi:.1f} 米   {m.sum():5d}  {hit[m].mean()*100:5.1f}%  {e[m].mean()*100:5.1f} cm")
for name, k, labels in [("左右 y (+左/-右)", 1, None), ("上下 z (+高/-低)", 2, None), ("前后 x (+远/-近)", 0, None)]:
    print(f"\n按 {name} 分组（相对标准点）:")
    for lo, hi in [(-2, -.6), (-.6, -.3), (-.3, .3), (.3, .6), (.6, 2)]:
        m = (d[:, k] >= lo) & (d[:, k] < hi)
        if m.sum(): print(f"  [{lo:+.1f}, {hi:+.1f}) 米  样本 {m.sum():5d}  命中率 {hit[m].mean()*100:5.1f}%  平均误差 {e[m].mean()*100:5.1f} cm")
EOF
```

实测 `forehand_new` 的结论：
- 离标准击球点 **0.6 米以内几乎全部能打到**（97.8–100%，误差 3–7 厘米）；超过 0.8 米只有 56%。
- 最难的是**太高的球**：比标准点高 0.6 米以上（离地约 1.7 米）命中率只有 5%，已经接近机器人加球拍能够到的极限。
- 其次是**往持拍手一侧太远的球**：右偏 0.6 米以上命中率 71%。参考动作是原地正手，机器人不会迈步去追球。
- 想打到更远的球，需要再录几段**带步法**的正手（例如向右跨一步再击球、后退一步再击球），分别登记到 motion_lib.py，放进同一个 TOML 一起训练。作者的 `tennis_only.toml` 就是这样做的（`two_step_forehand`、`stepback_forehand`）。评估和部署时，会自动选标准击球点离目标最近的那个动作。

原 README 中的其他回放和评估命令，写法同样改成 `python -m`：

```bash
uv run play-viser Mjlab-MultiTarget-Tracking-Flat-Unitree-G1 --wandb-run-path <entity>/mjlab/<run-id> --motion-config <toml>                  # viser 回放（这个命令名没有冲突，可以直接用）
uv run play-viser-motion-ranges Mjlab-MultiTarget-Tracking-Flat-Unitree-G1 --wandb-run-path <entity>/mjlab/<run-id> --motion-config <toml>    # 同时显示目标点的可达范围
uv run python -m tasknpoint_project.goal_cond_tracking.scripts.evaluate Mjlab-MultiTarget-Tracking-Flat-Unitree-G1 \
    --wandb-run-path <entity>/mjlab/<run-id> --motion-config <toml> \
    --target-x-min 0.4 --target-x-max 0.6 --target-y-min -2.0 --target-y-max 2.0 --target-z-min -0.4 --target-z-max 0.40   # 在目标范围内批量评估
```

---

## 附录 A：踩坑记录（必读）

| # | 现象 / 报错 | 原因 | 解决办法 |
|---|---|---|---|
| 坑1 | `the provided API key cannot access this resource` | 英文 README 里的 `demalenk-...`、`bwerner-...` 都是作者的**私有** wandb 数据 | 按第 6 节自己从视频生成数据 |
| 坑2 | 安装时出现 `launch-ros ... requires pyyaml` 等冲突 | `~/.bashrc` 里 source 了 ROS，`PYTHONPATH=/opt/ros/jazzy/...` 混进了 conda 环境 | 每次执行 `unset PYTHONPATH` |
| 坑3 | 安装日志里出现 `Requirement already satisfied: ... in /home/user/.local/lib/python3.12/site-packages` | 用户目录的旧包（例如 numpy 2.4）"冒充"已安装，导致版本错乱 | 安装和运行前执行 `export PYTHONNOUSERSITE=1`；如果已经装坏了：`conda env remove -n phmr_pt2.6 -y` 后重装 |
| 坑4 | `FileNotFoundError: .../PromptHMR/code/data/pretrain/camcalib_sa_biased_l2.ckpt` | `~/.bashrc` 里设置了 `AWS_DEFAULT_REGION`，`cam_calib.py` 误以为在 AWS 上，往路径里插了 `/code/` | 运行前执行 `unset AWS_DEFAULT_REGION` |
| 坑5 | `ValueError: Not all regular expressions are matched! ... racket_contact: []`；或 `--registry-name` 没有效果 | `uv run train` / `uv run play` 调用的是 **mjlab 自带的**同名命令，不是本项目的 | 改用 `uv run python -m tasknpoint_project.scripts.train`（play 同理）；数据来源用 `--motion-config` |
| 坑6 | `fetch_smplx.sh` 执行后 `smplx/` 目录不存在；或报 `gdown: 未找到命令` | 脚本不检查 wget 是否成功，失败后照样删除压缩包；gdown 只装在 phmr_pt2.6 环境里 | 按第 5.1 节手动下载并检查文件大小；执行脚本前先 `conda activate phmr_pt2.6` |
| 坑7 | `Too many users have viewed or downloaded this file recently` | Google Drive 对匿名下载有限流 | 按第 5.3 节：优先用官方镜像；gdown 单文件重试；再不行就用浏览器下载后 rsync 上传 |
| 坑8 | 训练时报 artifact 找不到 | 数据上传到了**默认 entity**（可能是团队名，例如 `xxx-itri`），不是用户名 | 用 `wandb.Api().default_entity` 查出来，填进 TOML 的 `train_prefix` |
| 坑9 | 上传后服务器上出现 `tasknpoint_uploads/tasknpoint_uploads/` | `scp -r 目录/` 会多套一层目录 | 改用 `rsync -avP 目录/ 服务器:~/目录/`（源路径末尾要有 `/`） |
| 坑10 | PromptHMR 首次运行时在 "HF Hub" 附近卡住好几分钟 | 要从 huggingface.co 下载约 1.9GB 模型，国内网络慢且时断时续 | 耐心等；它会继续下载（实测停了约 2 分钟后下载完成）。hf-mirror.com 在本环境中只会跳转回官方地址，没有帮助 |
| 坑11 | 训练出的动作比真人快或慢 | `extract_smpl_for_robo.py` 把帧率写死为 30 | `csv_to_npz.py --input-fps` 填视频的真实帧率（用 ffprobe 查） |
| 坑12 | 视频处理脚本 `install.sh` 显示 "completed"，但其实有包没装上 | 脚本没有 `set -e`，出错后继续执行 | 用第 4 节的 import 验证命令检查 |
| 坑13 | `uv run motion-set` 报 `Failed to spawn` | pyproject 里没有注册这个命令 | 用 `uv run python -m tasknpoint_project.motion_sets.motion_set <toml> --list` |
| 坑14 | 英文 README 中的 `all_motions.toml` 等配置找不到 | 仓库里没有这些文件 | 用 `ls .../motion_train_configs/` 查看实际存在的配置 |
| 坑15 | 重定向录像里只看到机器人的腿，或者机器人走出画面 | 无头模式下相机默认固定、距离太近 | 加 `--follow_camera --cam_distance_scale 3` |
| 坑16 | 多卡服务器上，训练或 PromptHMR 抢了别人的 GPU，或者 OOM | 默认使用 0 号卡；机器是多人共用的 | 先用 `nvidia-smi` 看空闲显存，再用 `CUDA_VISIBLE_DEVICES=<卡号>` 指定 |
| 坑17 | 重定向后的机器人**身体后仰、双脚悬空**，骨盆越走越高 | 原版 `construct_motion.py` 会把起始帧骨盆的完整 3D 朝向（包括横滚/俯仰）强制摆正，并把骨盆高度固定为 0.79 米。片段如果从屈膝、侧倾的姿势开始，整个地面就被转斜了：人横向移动 0.8 米，就会多出约 15 厘米的"上升" | 重定向时加 `--yaw_only_align`（第 6.5 节），然后用第 6.5 节的"检查 1"脚本确认脚踩在地上。**用错误数据训练出的策略没有意义，必须重新执行 csv_to_npz 再训练** |
| 坑18 | 执行 `pkill -f ...` 后自己的终端或脚本也退出了（exit code 144） | `pkill -f`/`pgrep -f` 会匹配完整命令行，执行这条命令的 `bash -c "..."` 里也包含同样的字符串 | 用 PID 文件 `kill $(cat logs/train_forehand_new.pid)`；需要按名字查找时，用 `[p]ython3` 这种方括号写法 |
| 坑19 | 训练日志文件长时间是空的，或者只有几行 | 输出重定向到文件时，Python 的 stdout 是块缓冲的 | 启动前执行 `export PYTHONUNBUFFERED=1` |
| 坑20 | 机器人**只有"拉手蓄力"，看不到击球**；手臂一直弯着 | 单目视频的深度歧义：球员**正对镜头**时，"手臂横向伸直"和"手臂朝镜头弯曲"在画面上几乎一样，PromptHMR 把手臂估计成弯的（实测第一段视频：真实手臂伸直时，重建出的肘角只有 93°，整段最大 125°） | 用第 6.3(b) 节的表检查"右肘最大角"，击球或随挥时应能到 140° 以上；用 6.3(d) 的叠加图确认。不合格就**换视频**：从侧面或斜 45° 拍 |
| 坑21 | 4K 视频处理很慢、很耗内存 | PromptHMR 会把所有帧读进内存 | 先用 ffmpeg 转成 1080p（第 6.1 节） |
| 坑22 | **把正手看成反手**（或者反过来） | 只看缩略图时，容易把"引拍"和"随挥"弄反 | 用第 6.3(b) 的速度表：**最快的那一段才是击球**。右手持拍时，从右往左快速挥是正手，从左往右是反手 |
| 坑23 | 录像里"先看到击球后的动作，然后才引拍" | 片段的起点落在上一拍的收拍中途，那一段速度很快 | 起点和终点都选速度最低的帧（第 6.3(c) 节），顺序应为 静止 → 引拍 → 击球 → 收拍 → 静止 |
| 坑24 | 骨架叠加图整体偏移、被拉长 | PromptHMR 内部把画面缩小后再处理（例如 1920 宽缩到 1592），相机内参是按缩小后的尺寸记录的 | 投影坐标乘以 `画面宽 / (2 × img_center_x)`（第 6.3(d) 脚本已经处理） |
| 坑25 | GMR 的 `--record_video` 录像里看不清手臂 | 相机在机器人左后方，而且不带球拍 | 用第 6.5 节"检查 2"的脚本，渲染带球拍的正面加侧面录像 |
| 坑26 | 渲染报 `Image height 600 > framebuffer height 480` | 训练模型 XML 里默认的离屏画布是 640×480 | 在代码里设置 `model.vis.global_.offwidth/offheight`（第 6.5 节脚本已经处理） |
| 坑27 | viser 网页显示 "WebGL is unavailable" | 本地浏览器没有开启 WebGL / GPU 加速 | 见第 9 节的 Chrome 设置步骤，或者改为在服务器上录 mp4 |
| 坑28 | `invalid choice '--video-length' for argument '--video'. Expected one of ('True', 'False')` | 本项目的命令行解析器（tyro）要求布尔参数显式写值 | 写成 `--video True` |
| 坑29 | 评估时报 `Warp CUDA error 2: out of memory` | 共享服务器上，别人的程序（例如 llama-server、sglang）把这张卡占满了 | 先用 `nvidia-smi` 找一张空闲显存 > 3GB 的卡，再改 `CUDA_VISIBLE_DEVICES` |
| 坑30 | 评估/回放报 `Could not find run ...`；训练日志里有 `run ... was previously created and deleted` | wandb 上的 run 在训练过程中被人删掉了（团队空间里其他人也能删）。**本地的模型存档不受影响** | 用 `--checkpoint-file` 指定本地的 `model_*.pt`（第 9、9.1 节） |
| 坑31 | 评估结果好得不真实（例如命中 99.9%），和录像里看到的不一致 | 原评估脚本统计整个 10 秒回合，第一拍之后目标点按训练分布重新随机，多数落在很容易打到的位置 | 加 `--first-motion-only True` |

## 附录 B：本次实测新增 / 修改的文件

| 文件 | 内容 |
|---|---|
| `tasknpoint_project/src/tasknpoint_project/motion_sets/motion_train_configs/forehand_new_only.toml` | 只练 `forehand_new` 一个正手动作的训练配置（**推荐**） |
| `tasknpoint_project/src/tasknpoint_project/motion_sets/motion_train_configs/backhand_only.toml` | 只练 `backhand_custom` 的训练配置（数据有坑20的问题，不推荐） |
| `tasknpoint_project/src/tasknpoint_project/motion_sets/motion_lib.py` | 新增 `forehand_new`（相位 0.75）和 `backhand_custom`（相位 0.63）两个条目，目标点都来自 probe 输出；原有条目没有改动 |
| `retarget/construct_motion.py` | 新增 `--yaw_only_align` 参数（坑17）；不加这个参数时行为与原版完全相同 |
| `tasknpoint_project/src/tasknpoint_project/scripts/play.py` | `--viewer` 新增 `none` 选项，用于在服务器上无头录制 mp4（第 9 节）；其他选项的行为没有变化 |
| `tasknpoint_project/src/tasknpoint_project/goal_cond_tracking/scripts/evaluate.py` | 新增 `--checkpoint-file`（使用本地模型，此时 `--wandb-run-path` 可以不填）、`--first-motion-only`（只统计第一拍，默认关闭）、`--per-env-file`（保存每个目标点的结果）；不加这些参数时行为与原版相同 |
| `README_CN.md` | 本文件 |

以下文件是运行时生成的，已被 `.gitignore` 忽略：`videos/`、`video_processing/human_pose_outputs/`、`retarget/retarget_inputs/`、`retarget/retarget_outputs/`、`retarget/videos/`、`tasknpoint_project/logs/`、`tasknpoint_project/wandb/`、`tasknpoint_project/artifacts/`。

## 许可证

本项目使用 [MIT 许可证](LICENSE.md)。子模块（PromptHMR、GMR、unitree_sdk2_wrapper）各自保留原有许可证。SMPL/SMPL-X 模型需要遵守其官网的许可协议。

## 引用

```
@article{tasknpoint2026,
    author    = {Werner, Blake and Demler, Ilona and Perona, Pietro and Ames, Aaron D.},
    title     = {TaskNPoint: How to Teach Your Humanoid to Hit a Backhand in Minutes},
    journal   = {https://arxiv.org/pdf/2606.26215},
    year      = {2026},
}
```
