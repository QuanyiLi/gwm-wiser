# 真机 Bring-up 手册（GWM × TiPToP，Franka + RealSense）

*本机踩点日期 2026-08-17，主机名 `zhiwei`，仓库分支 `hardware`，checkpoint `/home/quanyi/0810_gwm/checkpoint.pt`（1.4 GB，run-1 step 34000，即 G-19…G-32 全部 sim 结果所用的那个）。*

sim 侧状态（`plan.md` G-32）：scene6 四系统 280 trials 已收官 —— GWM-fusion 70/70、GWM-cam2 70/70、GWM-cam1 65/70、tiptop 64/70。pick / place 都跑通了，selection 用两段式（object 由 GWM 聚合，grasp 由 M2T2 confidence + closing-line gate）。真机是 M4，此前没有任何真机代码路径。

---

## 0. 本机现状

| 项 | 实测值 | 状态 |
|---|---|---|
| OS | Ubuntu 22.04.5 | ✅ tiptop 支持 |
| 内核 | `6.8.0-136-generic` `PREEMPT_DYNAMIC` | ✅ 不需要 RT —— Bamboo 跑在独立 NUC 上 |
| GPU | RTX 5090 32 GB，driver 580.126 | ✅ **`sm_120` 已实测通过**，见 §0.2 |
| 内存 | 62 GB | ✅ |
| 磁盘 | 2026-08-18 清理后剩 **~160 GB** | ✅ 够 |
| 相机 | RealSense D435 + D435i，均已安装 | ✅ 见 §0.4 |
| 网络 | `enp6s0` DOWN；`172.16.0.2` 不通 | ⬜ 机器人待上电（§2） |

`droid/README.md` 里的 `/root/code/gwm/...` 路径和 47 GB 的 pixi 环境是**旧机器**上的，本机一个都没有（全在 `.gitignore` 里），已全部重装 —— 重装过程中踩到的三个坑记在 `droid/README.md` 的 "Rig 2 — zhiwei" 一节。

### 0.0 Bring-up 进度（截至 2026-08-18）

**GPU 工作站软件栈已全部就绪并逐项实测通过。**

| 阶段 | 状态 |
|---|---|
| pixi 0.76.2 | ✅ |
| tiptop pixi env | ✅ 需 `SETUPTOOLS_SCM_PRETEND_VERSION_FOR_TIPTOP`，否则 editable install 静默失败 |
| cuRobo `b5fad1d`（GI-0 pin） | ✅ `sm_120` 原生编译 |
| cuTAMP v0.0.6 | ✅ |
| `cutamp-demo --motion_plan` 冒烟 | ✅ "Successful plan found!"，13 次运动规划 0.93 s |
| `gwm_tiptop` site-packages symlink | ✅ |
| RealSense preflight | ✅ 2/2 PASS，修了 IR 曝光 bug（§0.4a） |
| M2T2 env + `pointnet2_ops` | ✅ 用 `9.0+PTX` 编译（§0.2） |
| M2T2 权重 + server :8123 | ✅ 合成场景实测 2770 grasps，位置正确 |
| FoundationStereo env + 权重 + server :1234 | ✅ |
| **真实 RealSense → FoundationStereo 端到端深度** | ✅ 见下 |
| gwm venv + Qwen 权重 | ⬜ 下一步 |
| Robotiq 换装 | ✅ 用户 2026-08-18 完成 |
| 法兰立柱实测值 | ⬜ **待用户提供**（§0.1a 步骤 3） |
| 机器人上电 + 读 system version | ⬜ **阻塞 Bamboo**（§3.0） |
| Bamboo on NUC | ⬜ |

端到端深度实测（走 tiptop 自己的 `rs_infer_depth_async`，不是只 ping 服务）：

| | FoundationStereo | RealSense 自带深度 | 一致性 |
|---|---|---|---|
| D435 external | 88.3 % 有效 | 72.4 % | 中位 −3.3 mm，p90 62 mm |
| D435i wrist | 99.7 % 有效 | 77.5 % | 中位 +6.8 mm |

FoundationStereo 明显更密，符合预期（这正是 tiptop 不用 ASIC 深度的理由）。

### 0.1 rig 定义（2026-08-17 用户确认）

| 项 | 值 | 影响 |
|---|---|---|
| 控制机 | **独立 NUC（有 RT 内核）** | ✅ 就是 DROID 官方双机拓扑。Bamboo 装 NUC，`zhiwei` 当 GPU workstation，本机内核不用动 |
| 夹爪 | **Robotiq 2F-85**（用户有，决定从 Franka Hand 换过来） | ✅ 与 sim / GWM 训练语料一致，OOD 风险归零 |
| 机械臂 | **Franka Emika** → 待确认 Panda 还是 FR3 | 「Franka Emika」是公司名（Panda 时代品牌），FR3 由 Franka Robotics 出品 ⇒ **大概率 Panda**。§2 上电后在 Desk 里确认；两者切换只是改一个字符串 |
| GPU | RTX 5090，用户判断与 3090 无实质差异 | 按此推进；`sm_120` 若真的编译失败再回到 §0.2 的缓解手段 |

tiptop embodiment：**`panda_robotiq`**（FR3 则 `fr3_robotiq`）。

**换 Robotiq 的收益**（相对 Franka Hand，记录一下为什么这个决定重要）：

1. renderer 无需改动 —— `assets.py:build_welded_urdf` 本来就是「剥掉 Franka Hand、焊 Robotiq 2F-85」，`franka_renderer.py` 的 `MIMIC_MAP` / `DRIVER_RANGE_RAD=0.8` 也是为 2F-85 写的。**一行不用改。**
2. GWM 的 robot-only RAT 渲染帧与训练语料（MolmoAct2-DROID = Panda+Robotiq，MolmoBot = FR3+Robotiq）同分布，不必做「渲 Franka Hand vs 渲 Robotiq」的消融。
3. tiptop 的 `*_robotiq` embodiment **自带 DROID 腕相机的碰撞球**；裸 `panda`/`fr3` 没有建模，得自己往 cuTAMP 的 cuRobo config 里加，否则规划器不知道腕上那台 RealSense 的存在、会撞桌子。
4. `grasp_gate.py` 的 pad 几何（closing axis、open half-gap 0.0516、pad_z −0.030）是从 2F-85 的碰撞球自标定出来的，sim 上验证过；换手就得重标。

### 0.1a 换装 Robotiq 2F-85 的步骤（机械 + 固件 + 软件）

**机械**
1. 断电、E-stop 按下。拆 Franka Hand（法兰面 4×M5 + 电气插头）。
2. 装 Robotiq 耦合环（ISO 9409-1-50-4-M6）→ 2F-85 本体。DROID 标准是 Robotiq AGC-CPL-062-002 耦合板。
3. **量一次法兰立柱厚度**：`panda_link8` 法兰面到 Robotiq `base_link` 的距离（含耦合环）。sim 那台是 18.2 mm，gwm-wiser 默认 4 mm（对 MolmoBot 验证过）——**这是 per-rig 量，必须自己测**，填进 §6.3 的 URDF。
4. 通讯线：2F-85 是 RS-485，经 Robotiq USB 转换器接到**控制机（NUC）**，会出现 `/dev/ttyUSB0`。记下这个 tty，Bamboo 启动时要传。
5. 腕相机支架：DROID 的 RealSense/ZED 支架是装在耦合环上的，确认换手后腕相机的机械位姿变了 ⇒ **§5.5 的手眼标定必须重做**（本来也要做）。

**Franka Desk 固件侧（关键，漏了会频繁 reflex）**
6. Desk → Settings → **End-Effector**：按 DROID 文档填 2F-85 的惯性参数（质量 ~0.925 kg、质心、惯性张量）。参考 <https://droid-dataset.github.io/droid/software-setup/host-installation.html#updating-inertia-parameters-for-robotiq-gripper>。
7. 设置 TCP 变换（法兰 → 指尖）。
8. 重启控制器让参数生效。

**软件侧**
9. Bamboo：用默认模式启动（**不要**加 `--gripper_type franka`），它会额外拉起 Robotiq 的 gripper server：
   ```bash
   bash RunBambooController                     # 默认即 Robotiq
   bash RunBambooController -h                  # 看 tty / robot-ip 等参数
   ```
   自测 `python bamboo/examples/gripper.py`（**不带** `--gripper-type franka`），夹爪应开合。
10. tiptop：`tiptop-config` 时 embodiment 选 `panda_robotiq`（或 `fr3_robotiq`）。
11. 换手后重跑：§5.5 手眼标定 → §5.6 gripper mask（形状完全变了）→ §5.3 workspace 复核（2F-85 比 Franka Hand 长约 4 cm，q_capture 的可达性和桌面间隙都会变）。

### 0.2 RTX 5090 / `sm_120` —— 实测结论

用户判断 5090 与 3090 无实质差异，推进安装。**结论：对，但三个环境的处理方式不同**，记下来免得下次重装再踩。

| 环境 | torch | `sm_120` 原生？ | 处理 |
|---|---|---|---|
| tiptop | **2.7.1 / CUDA 12.9** | ✅ `arch_list` 含 `sm_120` | 无需处理；cuRobo 用 `TORCH_CUDA_ARCH_LIST="12.0"` 编译 |
| M2T2 | 2.4.1 / CUDA 12.0 | ❌ 最高 `sm_90` | `pointnet2_ops` 用 **`TORCH_CUDA_ARCH_LIST="9.0+PTX"`** 编译，靠驱动 JIT 前向兼容 |
| FoundationStereo | 2.4.1 / CUDA 12.0 | ❌ 最高 `sm_90` | 无自编扩展，直接靠 `compute_90` PTX JIT 运行 |

M2T2 用 `12.0` 会直接报 `ValueError: Unknown CUDA arch (12.0) or GPU not supported` —— 是 torch 2.4.1 的 `_get_cuda_arch_flags` 不认识这个 arch，不是 GPU 的问题。`9.0+PTX` 编出的 PTX 由驱动 JIT 成 sm_120 机器码，实测 `furthest_point_sample` 正常出结果。

（升级这两个环境的 torch 到 2.7 是另一条路，但会脱离上游 pixi.lock、且要重验模型数值，PTX 方案代价小得多。）

#### 0.2a PTX JIT 首调用延迟会撞穿 tiptop 的 10 s 超时

上面两个 PTX 环境有个运行时后果。实测 FoundationStereo 用真实 1280×720 IR 双目：

```
server 启动后第一次 /infer   33.3 s      <- 驱动 JIT 编译全部 kernel
之后每次 /infer               1.0 s
```

而 `tiptop/perception/foundation_stereo.py:102` 写死了 `aiohttp.ClientTimeout(total=10.0)` ⇒ **每次重启服务后的第一次采集必定 TimeoutError**（之后就再也不会）。`tiptop/` 保持 pristine，所以修复是运维层的 —— 起完服务、交给 tiptop 之前，先各发一个丢弃请求：

```bash
cd /home/quanyi/gwm-wiser
pixi run --manifest-path droid/tiptop/pixi.toml \
    python -m gwm_hardware.common.warm_servers --hand-serial <腕相机 s/n>
```

`droid/gwm_tiptop/warm_servers.py` 会查两个服务的 health，各打一发真实请求，报告耗时。预热后实测 M2T2 0.3 s、FoundationStereo 1.1 s。

**每次会话的固定开场白**（顺序不能反）：

```bash
# 1) 起服务
cd droid/M2T2 && pixi run server              # :8123
cd droid/FoundationStereo && pixi run server  # :1234
# 2) 相机 preflight
pixi run --manifest-path droid/tiptop/pixi.toml python -m gwm_hardware.common.rs_preflight
# 3) 预热服务
pixi run --manifest-path droid/tiptop/pixi.toml python -m gwm_hardware.common.warm_servers --hand-serial <s/n>
# 4) 这时才能跑 tiptop
```

好消息：32 GB 显存 > 3090 的 24 GB，G-8 那个「planner 与 gwm-server 不能共存」的顺序化限制有机会取消（tiptop planner ~6–10 GB + gwm-server ~19–20 GB ≈ 30 GB）。先按顺序化跑通再考虑共存。

### 0.3 磁盘 ✅ 已解决

需求估算：tiptop+cuRobo+cuTAMP 17 GB ＋ M2T2 13 GB ＋ FoundationStereo 20 GB ＋ gwm venv（torch+sapien）~15 GB ＋ Qwen3-VL-Embedding-8B 权重 ~17 GB ＋ SAM2 权重 ~1 GB ＋ 运行产物 ≈ **85–100 GB**。用户 2026-08-18 清掉了另一个用户目录占的 1.4 T，现剩 ~160 GB，够用。

### 0.4 相机清单（2026-08-17 实测）

两台都已物理安装并出流，**用户无需再配置**。用 tiptop 自己的 `RealsenseCamera` 类实测：

| | D435 `348522073586` | D435i `134322070906` |
|---|---|---|
| 位置（据实拍画面判断） | **external**（第三人称斜视桌面，黑背景布） | **wrist**（正俯视，画面下缘可见对称夹持件） |
| USB | 3.2 | 3.2 |
| 固件 | 5.15.1.55 | **5.12.15.50**（建议 5.17.0.10，偏旧，建议 `rs-fw-update` 升级） |
| K_color fx, fy | 905.6, 905.6 | 908.3, 906.7 |
| cx, cy | 635.4, 381.6 | 640.2, 356.7 |
| IR 基线 | 49.98 mm | 50.03 mm |
| RGB / IR×2 / depth | 全部正常 | 全部正常 |

两台挂在同一个 USB hub（`6-2.2` / `6-2.4`）上，压测过并发带宽：1280×720 color+IR×2 双机同开，**配对读取 28.1 fps**，无瓶颈。

⚠️ 换 Robotiq 后腕相机支架动过，上表的 wrist/external 分工要复核一次。

#### 0.4a IR 曝光 bug（已修，写进了 preflight）

两台相机开机状态都是 **IR 自动曝光 OFF、曝光锁死 40000 µs（最大值）**，导致 IR 双目 **83–87 % 像素饱和到 255**。

为什么致命：tiptop **不用** RealSense ASIC 自带的深度 —— `rs_camera.get_depth_estimator` 把 **IR 双目对**送给 FoundationStereo（`rs_infer_depth_async`）。IR 一饱和，投射器的散斑图案就没了，而白桌面上那是唯一的纹理源。下游全链（世界点云 → RANSAC 桌面 → DBSCAN 聚类 → M2T2 抓取 → cuTAMP 碰撞网格）都建在这个深度上。

开回自动曝光后实测：

```
D435   IR 饱和 87.1 % -> 1.8 %,  深度有效率 21.2 % -> 91.3 %
D435i  IR 饱和 84.6 % -> 1.7 %,  深度有效率 17.8 % -> 88.1 %
```

`RealsenseCamera.__init__` 完全不碰曝光，只开流，所以设备残留状态会**静默**带进每次运行。`tiptop/` 保持 pristine（G-18/G-21），因此修复以 pre-flight 形式落在 `droid/gwm_hardware/common/rs_preflight.py`，两条 arm 共用：

```bash
cd /home/quanyi/gwm-wiser
pixi run --manifest-path droid/tiptop/pixi.toml python -m gwm_hardware.common.rs_preflight
pixi run --manifest-path droid/tiptop/pixi.toml python -m gwm_hardware.common.rs_preflight --check  # 只报告不改
```

检查固件 / USB / 曝光 / IR 饱和度 / 深度有效率 / 内参 / 基线，需要时自动修，给 PASS/FAIL。**每次开工前跑一次**（相机重新插拔会回到默认状态）。

### 0.5 只有一台外部相机 ⇒ 拿不到 G-30 的双相机 fusion

sim 的最优配置是 `--cam external_cam,external_cam_2` 的双视角平均（G-30：两视角的 per-object 分数偏差相关性只有 r=+0.58，视角噪声和语义信号同量级）。本机 2 台 RealSense 必须一台当 wrist，所以只剩单一 external。

后果：**外部相机的摆位是一等大事**（G-29：换个视角 object accuracy 9/10 → 10/10）。摆位准则见 §6.1。如果后面发现打分不稳，最省事的补救是再加一台 RealSense 做第二 external —— 代码侧 `--cam` 已经支持逗号分隔的多相机融合，加一台就能直接用上 sim 的最佳配置。

---

## 1. 拓扑与端口

```
[控制机 (RT kernel)]                    [本机 zhiwei = GPU workstation]
  Franka 172.16.0.2  ──enet──  Bamboo     tiptop (pixi)      ── planner: cuRobo + cuTAMP
                               :5555 robot     │
                               :5559 gripper   ├─ M2T2 server            :8123
                                               ├─ FoundationStereo server :1234
                                               ├─ SAM2 (local, 仅 baseline arm 需要)
                                               └─ gwm-server (gwm venv)   :8901
                                        USB3: RealSense wrist + RealSense external
```

若没有第二台机器，Bamboo 和上面全部服务都在本机（前提：本机换 RT 内核）。

---

## 2. 阶段 A —— 机器人上电与联通（30 min）

1. 接好 `enp6s0` 到 Franka 控制器（不是 shop-floor 口，是 control 口），给该网卡配 `172.16.0.1/24` 静态地址。
2. 机器人上电、解 E-stop，浏览器开 `https://172.16.0.2/desk/`。
3. Desk 里：**Unlock joints** → **Activate FCI** → 模式设为 **Execution**。
4. 若是 Robotiq：确认 Desk 里已按 DROID 文档填过 Robotiq 的 **inertial parameters**（末端负载参数不对，libfranka 会频繁 reflex）。
5. 记下 Desk → Settings → Dashboard → Control 里的 **FCI 版本**，查 Franka 兼容表得到匹配的 **libfranka 版本**（Bamboo 安装时要输）。
6. 验证：`ping -c3 172.16.0.2` 通。

---

## 3. 阶段 B —— 控制机（NUC）装 Bamboo

### 3.0 先定 libfranka 版本 —— 这一步不能猜

Bamboo **不锁定** libfranka 版本，`InstallBambooController` 会交互式问你要装哪个，由**机器人的 system version** 决定。查法：Desk → Settings → Dashboard → Control 读出 robot system version，再查下表（Franka 官方 compatibility matrix，2026-08-18 核对）：

| libfranka | Robot System Version | Robot / Gripper Server |
|---|---|---|
| ≥ 0.18.0 | ≥ 5.9.0 | 10 / 3 |
| ≥ 0.15.0 | ≥ 5.7.2 | 9 / 3 |
| ≥ 0.14.1 | ≥ 5.7.0 | 8 / 3 |
| ≥ 0.13.3 | ≥ 5.5.0 | 7 / 3 |
| ≥ 0.10.0 | ≥ 5.2.0 | 6 / 3 |
| ≥ 0.9.1 | ≥ 4.2.1 | 5 / 3 |
| ≥ 0.8.0 | ≥ 4.0.0 | 4 / 3 |
| ≥ 0.7.1 | ≥ 3.0.0 | 3 / 3 |

**⚠️ Panda 用户注意（很可能就是本 rig）**：libfranka **0.10.0 是最后一个把 Panda 写进支持列表的版本**（CHANGELOG 里 0.10.0 写的是 "Panda system version >= 5.2.0"，从 **0.11.0 起改成 "Franka Research 3 system version >= 5.2.0"**，Panda 不再维护）。所以：

- **老 Panda（system version 4.x）→ libfranka 0.9.x**（通常 0.9.2）
- **升级过固件的 Panda（system 5.2.0）→ libfranka 0.10.0**，别再往上
- **FR3 → 按上表选**，一般落在 0.13.3–0.15.x

装新版 libfranka 连老 Panda 会直接握手失败，不是「先跑起来再说」的事。

**Pinocchio**：Bamboo README 说 libfranka ≥ 0.14.0 需要先装 Pinocchio（0.14.0 起用 Pinocchio 算动力学参数，0.18.0 起是硬依赖）。推论：
- 走 Panda 路线（0.9.x / 0.10.0）→ **不需要 Pinocchio**
- 走 FR3 路线（≥0.14.1）→ **必须先装 Pinocchio**，按 libfranka 的 dependency 说明来，再跑 Bamboo 安装脚本

所以 §0.1 里「Panda 还是 FR3」这个待确认项**阻塞本阶段** —— 上电读出 system version 之前不要开始装。

### 3.1 安装

在**有 RT 内核的 NUC** 上：

```bash
git clone https://github.com/chsahit/bamboo.git
cd bamboo
# 若走 FR3 / libfranka >= 0.14.0：先装 Pinocchio
bash InstallBambooController      # 交互式，输入 §3.0 定下的 libfranka 版本
# 脚本会本地编译 libfranka，不覆盖系统安装；会请求 sudo 加用户组
# 若提示加了用户组（realtime 等）→ 必须注销重登
```

Bamboo 支持 FR3 和 Panda 的 joint impedance control，夹爪支持 Robotiq 和原厂 Franka Hand。

### 3.2 启动与自测

```bash
bash RunBambooController          # 默认即 Robotiq（本 rig）；-h 看 tty / robot-ip 等参数
```

夹爪会开合一次表示激活成功。整场实验保持常驻。自测（**先清空机器人周围**，脚本不做碰撞检查）：

```bash
conda activate bamboo
python bamboo/examples/joint_trajectory.py
python bamboo/examples/gripper.py        # 本 rig 是 Robotiq，不加 --gripper-type
```

结束用 `bash RunBambooController stop`。

---

## 4. 阶段 C —— GPU 机装 tiptop 全栈（1–2 h，含编译）

先清磁盘（§0.3），再装 pixi：

```bash
curl -fsSL https://pixi.sh/install.sh | bash && source ~/.bashrc && pixi --version
```

**注意：本仓库已把 tiptop / M2T2 吸收成 monorepo 子目录，不要再按官方文档另外 clone。** FoundationStereo 仓库里没有（sim 用 GT depth），真机必须新装。

```bash
cd /home/quanyi/gwm-wiser/droid/tiptop
pixi install
pixi run setup-planners            # cuRobo + cuTAMP，5–15 min，5090 上大概率要调 §0.2
# 不用 pixi run install-zed（我们用 RealSense）

# 立刻冒烟，验证 sm_120：
pixi run cutamp-demo --motion_plan     # 应弹出 Rerun
pixi run tiptop-run -h
```

pyrealsense2 检查（`rs_camera.py` 是 lazy import，pixi 环境里要有）：

```bash
cd /home/quanyi/gwm-wiser/droid/tiptop && pixi run python -c "import pyrealsense2 as rs; print(rs.__version__)"
# 缺就装：pixi add --pypi pyrealsense2
```

M2T2：

```bash
cd /home/quanyi/gwm-wiser/droid/M2T2
pixi run setup
pixi run download-weights      # 需要 git-lfs，失败见 installation.md 的 troubleshooting
pixi run demo                  # meshcat http://127.0.0.1:7000/static/
```

FoundationStereo（**RealSense 走 IR 双目 → FoundationStereo 出深度**，`rs_infer_depth_async`，不是用 RS 自带的 stereo depth）：

```bash
cd /home/quanyi/gwm-wiser/droid   # 放这里，和其它子模块同级
git clone https://github.com/williamshen-nz/FoundationStereo.git
cd FoundationStereo
pixi run setup && pixi run download-checkpoints && pixi run demo
```

把 `droid/FoundationStereo/` 加进 `droid/.gitignore`（跟其它 `.pixi` 一样不入库），并在 `droid/README.md` 的表里补一行 provenance。

最后恢复 `gwm_tiptop` 的 site-packages 符号链接（**不要用 `.pth`，见 G-21**）：

```bash
ln -sfn /home/quanyi/gwm-wiser/droid/gwm_tiptop \
  "$(/home/quanyi/gwm-wiser/droid/tiptop/.pixi/envs/default/bin/python -c 'import site; print(site.getsitepackages()[0])')/gwm_tiptop"
```

⚠️ 跑 `python -m gwm_tiptop.*` 时 **cwd 必须是 `/home/quanyi/gwm-wiser`**（在 `droid/tiptop` 下会被嵌套的 `cutamp/` 遮蔽）。

---

## 5. 阶段 D —— tiptop 配置与标定（官方流程，先把 baseline 跑通）

这就是你说的「先按 tiptop setting 搭起来」。**这一整阶段不碰 GWM**，目标是让原版 tiptop（Gemini + SAM2）在你的桌面上能完成一次 pick-and-place。

### 5.1 配置

```bash
cd /home/quanyi/gwm-wiser/droid/tiptop && pixi shell
tiptop-config
```

填：embodiment（`fr3_robotiq`/`panda_robotiq`/`fr3`/`panda`，见 §0.1-2/3）、Bamboo 主机 IP 与端口 5555/5559、相机序列号、M2T2 `http://localhost:8123`、FoundationStereo `http://localhost:1234`。

然后手改 `tiptop/config/tiptop.yml`：

```yaml
cameras:
  hand:     { serial: "<腕上那台的 s/n>", type: realsense }
  external: { serial: "<外部那台的 s/n>", type: realsense }
robot:
  time_dilation_factor: 0.2      # 先 0.2，线缆理顺后再往上，最高别超 0.6
```

（D435 `348522073586` 和 D435i `134322070906` 二选一当 wrist —— **建议 D435i 当 external**，因为它有 IMU 便于核对水平，且 wrist 端体积/线缆更敏感。最终以你实际装在腕上的那台为准。）

Gemini key（baseline arm 需要，GWM arm 不需要）：

```bash
echo 'GOOGLE_API_KEY=<your-key>' > /home/quanyi/gwm-wiser/droid/tiptop/.env
chmod 600 /home/quanyi/gwm-wiser/droid/tiptop/.env    # 已 gitignore
```

### 5.2 连通性

```bash
get-joint-positions      # 通 = Bamboo 链路 OK
viz-gripper-cam          # q 退出
```

### 5.3 工作空间障碍物（安全关键）

编辑 `tiptop/workspace.py` 里 `fr3_workspace`，用 cuboid 描述：桌面、墙、相机支架、显示器、任何 keep-out 区。**宁可画大不要画小。**

```bash
python tiptop/workspace.py     # 可视化，反复迭代到合理
```

### 5.4 capture 位姿

```bash
go-to-capture            # ⚠️ 会动！手放 E-stop
viz-gripper-cam          # 看腕相机是否覆盖整个操作区
```
不满意：Desk 切 **Programming mode** → 手动摆到俯视全景位 → `get-joint-positions` → 抄进 `robot.q_capture` → 切回 **Execution mode**。TiPToP 全程只用这一帧，覆盖不全后面全废。

### 5.5 腕相机手眼标定

用 DROID 的 charuco 板；`tiptop/scripts/calibrate_wrist_cam.py` 里的参数是 **14×9 格、checker 20 mm、marker 15 mm、DICT_5X5_100** —— 板子不一样先改这里。

```bash
calibrate-wrist-cam
# Programming mode 手动把板子摆到左目画面中央、距离 30–60 cm → 切回 Execution → 按 y
# 2–3 min 后写入 tiptop/config/assets/calibration_info.json（确认文件真的更新了）
```

### 5.6 夹爪 mask

```bash
compute-gripper-mask     # Gemini 自动检测，满意按 y
# 效果差就手绘：
paint-gripper-mask
```

### 5.7 验收

```bash
viz-calibration          # Rerun：相机坐标轴要贴合夹爪，点云形状正确
```

### 5.8 baseline 跑一次（GI-1 的真机版）

三个终端：

```bash
# T1
cd /home/quanyi/gwm-wiser/droid/M2T2 && pixi run server
# T2
cd /home/quanyi/gwm-wiser/droid/FoundationStereo && pixi run server
# T3
cd /home/quanyi/gwm-wiser/droid/tiptop && source .env && pixi shell
tiptop-run --cutamp-visualize --no-execute-plan     # 先不执行，只看 Rerun 里的运动规划
tiptop-run                                          # 满意后真执行，手放 E-stop
```

**阶段 D 的退出条件**：原版 tiptop 在你的场景上连续 3 次「put the X in the Y」成功。达不到就不要往下走 —— 后面 GWM arm 复用的是同一套感知/规划/执行，baseline 不稳的话根因分不清。

---

## 6. 阶段 E —— GWM 侧的真机新增件（sim 里没有的部分）

这一阶段是**新工作**，不是照文档抄。三件事：外部相机外参、renderer overlay gate、真机 driver。

### 6.1 外部相机的摆放与外参标定（新代码，~100 行）

GWM 打分需要 `world_from_cam`（robot base 系下的外参）+ K + RGB，见 `score_client.py:174-182` 读的 h5 结构：`{cam}/rgb`、`{cam}/intrinsic_matrix`、`{cam}/pos_w`、`{cam}/quat_w_ros`。**tiptop 没有外部相机外参标定脚本，要自己写。**

**摆位准则**（从 G-29/G-30 学到的）：
- 参考 droid-sim 的几何：base 系 `(0.05, ±0.57, 0.66) m`，光轴下倾 ~37°，1280×720 —— 训练/评测语料就是这个 rig shape。
- 必须能看清**整只手臂 + 夹爪 + 全部候选物体**（robot-only 渲染帧要和真图对齐，手臂被切掉就没法打分）。
- 避开夹爪自遮挡：G-29 的 `yellow` 失败纯粹是目标落在夹爪阴影里，换个视角就修好了。
- 两侧各试一次，用 §6.2 的 overlay 覆盖率和一组已知指令的打分 margin 选，**并把「视角是事后挑的」这件事记进 ledger**（G-29 的诚实声明）。

**标定方法**（不需要新硬件）：腕相机已经手眼标定过，所以
1. charuco 板平放桌面不动；
2. `go-to-capture`，用腕相机 solvePnP 得 `T_wristcam_from_board`，链上 FK 与手眼外参 → `T_base_from_board`；
3. 同一块板用外部相机 solvePnP 得 `T_extcam_from_board`；
4. `T_base_from_extcam = T_base_from_board @ inv(T_extcam_from_board)`；
5. 多摆几个板位取平均，残差 > 5 mm 就重来。

产出写成 `external_obs.h5`（键名同上），复用 `tiptop/scripts/calibrate_wrist_cam.py` 里的 `CharucoDetector`。

### 6.2 Renderer overlay gate（硬门槛，等价于 GI-2）

在 sim 里这一关是 FK vs `body_pos_w` 对到 0.0 mm 才放行的。真机必须重做一遍，否则 RAT 帧里的机器人和真图错位，打分全是噪声：

```bash
cd /home/quanyi/gwm-wiser
# 用 real_data_train.renderer.FrankaRobotRenderer 以外部相机的 K/c2w 渲染当前 q，
# 叠到外部相机真图上（参考 gwm_tiptop/validate_renderer_overlay.py 与 overlays/ 里的 sim 样例）
```

判据：手臂各连杆边缘对齐、无重影、robot pixel coverage 与 sim 同量级（13 % 左右）。不过关就依次查：外参（§6.1）→ 法兰立柱（§6.3）→ URDF 选型（§0.1-2/3）。

### 6.3 URDF 与法兰立柱

`droid/gwm_tiptop/assets/panda_robotiq_droidsim.urdf` 里的 **18.2 mm** 立柱是 droid-sim 那台机器测出来的（gwm-wiser 默认值是 4 mm，对 MolmoBot 验证过）—— **per-rig，真机要重量一次**：量 `link8` 法兰面到 Robotiq `base_link` 的距离（含耦合环），改 URDF，再回 §6.2 复验。

### 6.4 真机 driver（新代码）

现有 driver 全是 sim 的 h5 离线链路：`propose_from_h5` → `score_client` → `policy_server`（说 tiptop websocket 协议，喂 droid-sim）。真机上 `tiptop_run.py` 自己管采集和执行，**中间缺一个把两半接起来的脚本**，建议叫 `gwm_tiptop/run_real.py`：

```
1. 复用 tiptop_run.capture_live_observation() 采腕相机 RGB + FoundationStereo depth + K + FK 位姿
   → 存成 propose_from_h5 要的 h5（键：rgb / depth / intrinsic_matrix / pos_w / quat_w_ros / q_init）
   注意 load_h5_observation 里那个 -15 mm 的 grasp-frame 修正是 droid-sim websocket client 的约定，
   真机要确认是否同样适用（这是个已知的 magic number，见 magic_numbers.md）
2. 同时抓外部相机一帧 → external_obs.h5（§6.1 的键）
3. python -m gwm_tiptop.propose_from_h5  --h5-path ... --output-dir ... --k-total 16
4. python -m gwm_tiptop.score_client --proposals-dir ... --external-h5 ... \
       --instruction "<verbatim task>" --cam <你的外部相机名> --rat-scale 3.0 --object-score mean
5. python -m gwm_tiptop.grasp_gate --apply    （两段式的第二段，G-27/G-28）
6. 读 winner_*.json（serialize_plan 格式）→ tiptop.execute_plan.execute_cutamp_plan(plan, client)
```

工作量不大（两半都有现成实现），但 3 和 4 分属两个 Python 环境（tiptop pixi vs gwm venv），按 G-8 顺序化跑：propose 完先放掉 planner 显存，再起 gwm-server 打分。5090 的 32 GB 有机会共存，但**第一次跑请老实顺序化**。

---

## 7. 阶段 F —— gwm-server（本机 GWM 环境）

```bash
cd /home/quanyi/gwm-wiser
python3 -m venv .venv          # 或 uv venv；需要 Python ≥3.11，本机系统 python 是 3.10
                               # ⇒ 建议用 uv/conda 装一个 3.11+ 再建 venv
.venv/bin/pip install -e '.[gwm+wiser]'      # 会带上 transformers==4.57.6 + mani_skill(sapien) + lerobot
```

⚠️ **绝对不要**把这个 env 和 tiptop 的 pixi env 合并（D9：`transformers==4.57.6` 与 tiptop 的 py3.12 栈冲突，进程边界是设计的一部分）。
⚠️ torch 要装 cu128 轮子，否则 5090 上跑不了（§0.2）。

首次会拉 `Qwen/Qwen3-VL-Embedding-8B`（~17 GB，先确认磁盘）。

启动：

```bash
cd /home/quanyi/gwm-wiser && .venv/bin/python -m droid.server.gwm_server \
    --backend gwm \
    --urdf droid/gwm_tiptop/assets/<你的真机 urdf> \
    --arm <panda|fr3> \
    --ckpt /home/quanyi/0810_gwm/checkpoint.pt \
    --port 8901
```

先用 `--backend dummy` 把 HTTP 链路和渲染 seam 跑通（不吃 20 GB 显存），再换 `gwm`。

**运维教训（G-25，踩过坑）**：关服务用端口，别 kill `$!` —— `fuser -k 8901/tcp`。曾经因为 kill 掉 wrapper 子 shell 而留下 20 GB 孤儿进程，把后续任务饿死一小时。

---

## 8. 阶段 G —— 第一次真机 GWM pick

1. 桌面摆一个**和 scene6 同构**的场景：3–5 个物体，含一个「明显目标 + 若干干扰物」，物体间距 ≥ 5 cm（DBSCAN eps 1.5 cm，太近会并簇）。
2. 跑 §6.4 的 1–4 步，**先只到打分**，人工检查：
   - `proposals_index.json` 里每个感知簇都有候选（真机点云比 sim 脏，`perception_geometric` 的 table plane / cluster merge 阈值可能要调，见 plan.md 的 GI-3 lessons）；
   - `cluster_viz.png` 上的分簇要和肉眼一致；
   - `scores_*.json` 的 `selected_target` 是不是对的物体；margin 有多大（sim 上 fusion 是 +0.0148…+0.0687，单相机会更抖）。
3. 打分对了再执行，`time_dilation_factor` 保持 0.2，手放 E-stop。
4. 同一条指令重复 5 次（真机没有 sim 的 determinism，这里才是真的在测 robustness）。

**头号预期风险**：G-24/G-26 反复出现的「retrieval 只打语义对齐、打不出抓取鲁棒性」。sim 里靠 stage-2（M2T2 confidence）+ closing-line gate 修好了，但 `grasp_gate.py` 里 `MIN_SLAB_PTS=150`（绝对点数，rig 相关）和 `MIN_THICKNESS=15 mm`（实心物体偏置）都是 class D magic number，**真机点云密度不同，这两个阈值一定要重标**（见 `magic_numbers.md` 的 grasp_gate 章节）。

---

## 9. 阶段 H —— A/B 协议

复刻 G-30/G-32 的四系统对比，真机版建议先砍到两臂：

| Arm | 组成 |
|---|---|
| `tiptop` | 原版：Gemini + SAM2 + cuTAMP，每 trial 重新规划 |
| `gwm_tiptop` | 几何感知 + M2T2 + cuTAMP 提案 → GWM 打分 → grasp gate → 执行 |

同一批场景、同一批 verbatim 指令、每条 ≥5 trials，人工判成功（真机没有 sim 的 rigid-body ground truth；建议同时录外部相机视频）。指令集直接搬 `refer6_tasks.sh` 的 10 条 referring expression + `place_tasks.sh` 的 4 条 destination RE，物体换成你手边的。

---

## 10. 时间估计

| 阶段 | 时间 | 阻塞点 |
|---|---|---|
| §0 清磁盘 + 确认 RT 内核/夹爪/臂型 | 0.5 d | **你来定** |
| A 机器人上电联通 | 0.5 h | |
| B Bamboo | 0.5 d（含可能的 RT 内核安装） | |
| C tiptop 全栈 | 0.5–1.5 d | **5090 sm_120 编译** |
| D 标定 + baseline 跑通 | 1 d | |
| E 外参标定 + overlay gate + run_real.py | 1.5–2 d | 全是新代码 |
| F gwm-server | 0.5 d | Qwen 权重下载 |
| G 首次 GWM pick + 阈值重标 | 1 d | |
| H A/B | 1–2 d | |

合计约 **7–9 个工作日**，其中 §0 的三个确认和 §0.2 的 5090 编译是最可能失控的两处。
