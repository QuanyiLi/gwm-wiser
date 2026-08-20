# GWM arm 真机手册

*建于 2026-08-19。baseline tiptop 已在 2026-08-18 跑通（`tiptop-modifications.md` 末尾四次成功 pick/place）。本文覆盖 `hardware-bringup.md` §6 那一整段「GWM 侧的真机新增件」，并记录哪些已经做完、哪些还卡在机器人上。*

## 0. 一句话现状

**除了「需要机器人动起来」的三件事，GWM arm 的真机链路已经全部搭好并在真实 rig 数据上跑通过。**

| 状态 | 内容 |
|---|---|
| ✅ 全部跑通 | 代码分层、观测 h5、几何感知、cuTAMP 提案、grasp gate、真 GWM 打分、调试可视化、2F-140 渲染模型、**外参标定**、**overlay gate**、**真机执行** |
| ⚠️ 已接线未验证 | **placing**（sim 版假设焊接方块 + sim 料箱，硬件上没跑过） |

**2026-08-19：第一次真机 GWM 抓取成功。** 指令 `grasping the object between the
tomato and the blue cup`（关系指代，不是点名），选中中间那个盒子，margin +0.0162，
gate 通过，5.73 s 抓住。

入口只有一个：

```bash
./droid/gwm_hardware/gwm_arm/run.sh            # 只提案打分，不动
./droid/gwm_hardware/gwm_arm/run.sh --execute  # 每一次动作仍需确认
```

**一共只有两类指令,由夹爪状态决定**（`get_gripper_state()` 实测，不是记忆也不是推断）：

| 夹爪 | 提案 | 执行 |
|---|---|---|
| 空 | PICK（`gwm_arm.propose`） | 按打分结果抓取 |
| 夹着东西 | PLACE（`gwm_tiptop.place_propose`） | 放置 → **张开 → 回 home（先抬 z）** |

**打分与 droid-sim 完全一致**（候选、两段式选择、数字都一样，sim 对比因此依然成立）；
**执行故意不一致**：sim 的 place 结束时方块还夹着，因为那里的"抓取"是焊接、松开是空操作
（G-25）；真机上夹着就等于没放下。多出来的两步是"已经放下了"的确定性后果，不是选择，
所以不需要候选也不需要分数。`gwm_tiptop/` 一行没动，分歧留在硬件侧。

---

## 1. 代码分层（和 tiptop arm 怎么分开的）

用户要求两条 arm 互不干扰、共用部分抽到公共层。落地成 `gwm_hardware/` 下的三个子包：

```
common/       rig 本身：机器人模型、标定、相机、workspace、五个 tiptop 树 installer
tiptop_arm/   baseline TiPToP（Gemini + SAM2 + cuTAMP）—— A/B 对照臂
gwm_arm/      GWM x TiPToP —— 被测方法
```

规矩只有一条：**`gwm_arm` 只 import `common`，两条 arm 之间零 import**。要被两边用的东西往 `common` 沉，不允许横向伸手。

**方法本身仍然在 `droid/gwm_tiptop/`，和 droid-sim 共用，没有 fork。** `gwm_arm/` 只放「sim 里不需要、真机上必须有」的那部分管道：实时采集、外部相机外参、overlay gate、执行、调试视图。

原来 `gwm_hardware/` 下平铺的 17 个文件都用 `git mv` 移进了 `common/`（3 个进 `tiptop_arm/`），路径常量集中到 `common/paths.py`，仓库里所有 `gwm_hardware.xxx` 引用（含 tiptop 树里三处 installer 打的补丁）同步改成 `gwm_hardware.common.xxx`。`assets/` 和 `config/` 留在包根，因为两条 arm 和 installer 都读它们。

验证：`install_2f140_cutamp --verify` 通过（2F-140 spheres z reach 213.2 mm）、`tiptop-run -h` 正常、`tiptop.yml` 符号链接完好、`workspace_cuboids()` 仍派发到 `zhiwei_workspace` 的 5 个 cuboid。

---

## 2. 共享代码改了什么（三处，全部向后兼容）

真机需要三件 sim 里不需要的东西。三处改动都做成「默认行为逐位不变」，droid-sim 的复现性不受影响。

### 2.1 `gwm_tiptop/propose_from_h5.py` —— −15 mm 外参修正可关掉

`load_h5_observation` 原本无条件把相机降 15 mm，对齐 droid-sim websocket client 的行为（`magic_numbers.md` #F）。那是**那个 client 的性质**，不是 pipeline 的：真机 `world_from_cam = FK × hand-eye`，本来就对，再降 15 mm 会把整片点云压穿桌面。

现在读 h5 里可选的 `extrinsics_z_correction`，缺省仍是 −0.015。sim 的 h5 没这个键 ⇒ 行为逐位不变；`gwm_arm.capture` 写的 h5 里显式写 0.0。

### 2.2 `gwm_tiptop/perception_geometric.py` —— 沿桌面法向切，而不是沿世界 z

**这是本轮在真机数据上抓到的第一个真问题。** 见 §3。新增 `cluster_objects(..., use_plane_normal=False)`，默认关。

回归实测（sim `smoke_test.h5`）：默认路径与改动前的 mask **逐点相同**（204044 点，完全一致）；打开开关后在 sim 上只差 164 点（0.018 %，因为 sim 桌面本来就只倾斜 0.079°）。

### 2.3 `real_data_train/renderer/franka_renderer.py` —— 认 URDF 自带的 mimic 链

渲染器原本硬编码 2F-85 的 `MIMIC_MAP`（ManiSkill 那个 URDF 没有 mimic 标签，六个独立 revolute）。本 rig 是 2F-140，它的 URDF 有 mimic。新增 `parse_mimic_chain(urdf)`：**文件里有 mimic 就照文件来，没有就走原来的 2F-85 分支**。2F-85 路径一行没动。

另外 `droid/server/gwm_server.py` 不用改——`--urdf` 本来就是参数。

---

## 3. 真机数据上抓到的：桌面在感知里是斜的

拿 2026-08-18 baseline 那次 `pick the blue cup` 的采集，直接跑 GWM 提案器，第一版结果是 **10 个候选、object_0 完全拿不到候选**，而且 `clusters.png` 上 Chocopie 盒子右边一大片空桌面被划进了 object_0。

量出来的根因：

```
拟合桌面法向  [-0.0086, 0.0496, 0.9987]   倾角 2.88°
垂直残差 rms  1.87 mm          <- 确实是个平面，不是噪声
inlier 世界 z  0.0361 .. 0.0839  跨度 47.8 mm
```

2.88° 就是 `tiptop-modifications.md` 末尾那个未解决的 hand-eye 旋转残差。平面本身很平（1.87 mm），但在 0.85 m 的采集足迹上，它自己的高度在世界 z 里就散开 **47.8 mm**——而「物体离开桌面」的判据只有 15 mm。于是桌面高的那一端比切面高出 24 mm，整片桌子变成一个假物体。

修法是沿**拟合平面的法向**量高度，而不是沿世界 z。水平桌面上两者完全等价，斜桌面上后者才是对的。

效果（同一份采集，只改这一个开关）：

| | 候选数 | 分布 | 假簇 |
|---|---|---|---|
| 世界 z 切（sim 默认） | 10 / 16 | object_0 拿到 **0** | 桌面右侧混入 object_0 |
| 平面法向切 | **16 / 16** | 6 / 5 / 5 | 无 |

`gwm_arm/propose.py` 默认开启；`--horizontal-cut` 可以切回 sim 行为做对照。

---

## 4. 已跑通的链路（robot 关着）

```bash
cd /home/quanyi/gwm-wiser
export PATH="$HOME/.pixi/bin:$PATH"
D=droid/gwm_hardware/runs/replay_2126_bluecup

# 1) 把 baseline 存下来的一次真机采集重放成提案器要的 h5
pixi run --manifest-path droid/tiptop/pixi.toml python -m gwm_hardware.gwm_arm.capture \
    replay droid/tiptop/tiptop_outputs/eval/2026-08-18_21-23-50 --out-dir $D

# 2) 提案（几何感知 + M2T2 + cuTAMP + cuRobo）
pixi run --manifest-path droid/tiptop/pixi.toml python -m gwm_hardware.gwm_arm.propose \
    --h5-path $D/wrist_obs.h5 --output-dir $D/proposals --k-total 16

# 3) 打分 + gate + 可视化（gwm-server 用 dummy backend 起着）
pixi run --manifest-path droid/tiptop/pixi.toml python -m gwm_hardware.gwm_arm.run_real \
    --run-dir $D --instruction "pick up the blue cup" --stages score,gate,viz
```

产出：`proposals/clusters.png`、16 个 `plan_*.json`、`scores_*.json`、`gate.json`、
`score_overlay_*.png`。

**这一轮的分数没有意义**：gwm-server 跑的是 dummy backend（按轨迹哈希打分，设计如此），
而且 external 视角是用 `capture wrist-as-external` 从腕相机伪装的——相机位姿是真的（FK ×
hand-eye），但腕视角不是打分视角。这一步验证的是管道，不是选择质量，文件自己的 attrs 里
也写了这句话。

### grasp gate 在真机点云上的读数

pad 几何自标定出来是对的：2F-140 open half-gap **0.0691 m**（2F-85 是 0.0516），闭合轴 `[-0.999, -0.051, 0]`。

```
object_0 (Chocopie 盒)  n=7457  thick=78.1  center= 7.9  ortho=15.4   6/6 FAIL
object_1 (蓝杯)         n≈1700  thick=14-26 center=1-18  ortho=0-3    3/5 PASS
object_2 (番茄)         n≈7000  thick=53-60 center=1-10  ortho=1-13   3/5 PASS
```

两件值得记的事：

1. **`MIN_SLAB_PTS = 150` 在本 rig 上完全不起作用。** 真机 slab 是 1500–8400 点，比 sim 密一到两个数量级（FoundationStereo 1280×720 vs sim 的降采样云）。它是 class D 的绝对点数阈值（`magic_numbers.md` #8），本来就说明了要按 rig 重标——但**没有执行失败证据之前不动它**，先记下来。`--min-slab-pts` 已经暴露成命令行参数。
2. **Chocopie 盒的 6 个候选四项指标完全相同**（M2T2 confidence 也都是 0.036）。M2T2 在这个扁盒子上只给出一族抓取，SE(3) FPS 只能返回近重复，于是 16 的预算里有 6 个是同一个候选。这会浪费预算，也让 `--object-score mean` 在这个物体上退化成单点。真机场景选物体时值得避开这类扁平大盒，或者把这一条作为 A/B 的已知限制记下来。

---

## 5. 还卡在机器人上的三件事，以及 Bamboo 一起来就怎么做

### 5.0 相机：先单相机，再考虑融合

rig 上有三台 RealSense，config 里现在三台都登记了：

| key | s/n | 角色 |
|---|---|---|
| `cameras.hand` | 035422072950 | 腕相机，**唯一的规划视角**（深度只来自它） |
| `cameras.external` | 348522073586 | 第三人称之一，D435，侧视桌面 |
| `cameras.external_2` | 134322070906 | 第三人称之二，D435i，**正对机器人** |

在 h5 和 `--cam` 里它们叫 `external_cam` / `external_cam_2`——和 droid-sim 同名，
所以 sim 上实测最优的那个字符串 `--cam external_cam,external_cam_2`（G-30 双视角融合）
在这里可以原样用。

**bring-up 期间默认单相机 `external_cam`**（侧视那台 D435）：一台相机 = 一个外参要标、
一个 overlay gate 要看、分数不对时只有一个地方可以怪。两台都标完、都过 gate 之后，
升级只是加一个逗号。

选侧视这台而不是正对机器人那台，只有一个理由，而且是决定性的：**正对那台冲着窗户，
而 RGB 是打分器唯一拿到的东西**。侧视这台背后是黑布，手臂、夹爪、整个桌面都已经在画面里
（`runs/camera_snapshots/`）。正对那台留作第二视角，等窗户问题解决再说。

⚠️ **`rs_preflight` 判 `external_2` FAIL 与打分无关。** 那条判据看的是 IR 双目饱和度，
服务于 FoundationStereo 的深度；打分相机只出 RGB，pipeline 从不问它要深度。**唯一的深度来自腕相机。**

⚠️ **`external_2` 现在还不能当打分视角**（2026-08-19 实拍）：它把手臂拍得很完整，但**桌面和
物体全在画面下方之外**，而且正对窗户过曝。要用它得先按 §5.0a 重新对准并解决逆光。

### 5.0a 对准打分相机

```bash
# 手臂必须先到 capture 位——那是每一集拍场景照的姿态，夹爪就在那里
pixi run --manifest-path droid/tiptop/pixi.toml go-to-capture
pixi run --manifest-path droid/tiptop/pixi.toml python -m gwm_hardware.common.aim_camera \
    --serial 134322070906
```

画面上有引导框，判据（`aim_camera` 的 docstring 就是这四条）：整只手臂从底座到指尖在框内、
每个候选物体可见且不被夹爪挡住、没有强逆光、桌面占画面相当一部分。`q` 退出，`s` 存图。

### 5.0b 打分相机的画幅裁剪：只保留左 2/3（2026-08-20）

`external_cam` 重新架设之后，采集时**裁掉画面右侧 1/3，只保留左 2/3**
（`tiptop.yml` 里 `cameras.external.keep_left_frac: 0.667`，由
`gwm_arm/capture.py` 的 `external_camera_crops()` 在写 `external_obs.h5` 之前执行）。

**为什么这样架、这样裁**：新机位把机器人的感知场更多地摊进画面，图像里的左/右
和工作区的左/右清晰对应——GWM 的空间 grounding 吃的就是这个；右侧 1/3 没有
这些信息，留着只是稀释画幅。

**为什么敢裁**：模型对相机位姿是鲁棒的。到目前为止试过的每一个机位（腕相机冒充
外视角、旧侧视机位、本次新机位）它都能把指令 ground 到具体的物体/放置属性上
（G-34 三指令 3/3、G-46 换机位后语言项照样选对对象），所以换机位 + 裁剪不碰
它的判别能力。

**机制上为什么便宜**：只裁右边，像素原点不动，**K 原样有效**；下游（gwm-server
渲染、overlay gate、viz）的宽度都取自图像本身，采集处一刀、全链路自动一致。
恢复全幅 = 删掉 yml 里那一行。裁剪只作用于打分照片——`extcam_calib` 标定和
`aim_camera` 预览用的都是原始全幅帧，腕相机（深度）完全不涉及。

⚠️ 换机位后 lift 阈值的现状：此机位 overlay gate 的 `edge_lift` 只有 ~0.075
（阈值 0.1），是轮廓两侧对比度的物理上限（黑爪衬黑布），**不是外参错**——
抓外参错的 `perturb_margin` 通过（+0.069/0.05），标定 pooled 重投影 0.44 px、
板 origin 平均差 1.3 mm。跑 gate 时用 `--min-lift 0.06`。

### 5.1 外部相机外参（新代码，`gwm_arm/extcam_calib.py`）

GWM 从第三人称相机打分，需要它在 base 系下的 `world_from_cam`。tiptop 只标腕相机。方法（`hardware-bringup.md` §6.1）：腕相机已经手眼标定过，用它把 base 系带到桌上的 charuco 板，外部相机再读同一块板。

拆成三条命令，**只有 `shoot` 碰机器人，而且只读关节角、绝不发运动指令**：

```bash
P="pixi run --manifest-path droid/tiptop/pixi.toml python"

# 手臂停在 capture 位，板子放桌面
$P -m gwm_hardware.gwm_arm.extcam_calib shoot --shots-dir droid/gwm_hardware/runs/extcal --n 6
# 每一 shot 挪一次板子；脚本当场逐相机报角点数和重投影误差

$P -m gwm_hardware.gwm_arm.extcam_calib solve  --shots-dir droid/gwm_hardware/runs/extcal
$P -m gwm_hardware.gwm_arm.extcam_calib check  --shots-dir droid/gwm_hardware/runs/extcal
```

`shoot` 一次把**腕相机和全部第三人称相机**都拍下来，所以一次摆板同时标定所有视角，而且它们
共用同一个腕相机解。**腕相机必须看到板子**（base 系是它带过去的）；某一台第三人称相机这一
shot 没看到板子不影响别的相机——各相机的解是独立的，`solve` 也会逐相机分别判定，
一台 PASS 一台 FAIL 时只安装 PASS 的那台。

判据：同一台相机在多次摆放之间的平移散布 **≤ 5 mm**，否则 FAIL 且拒绝安装。通过后写进
`common/config/extcam_calib.json`（`solve` 自动安装，`--no-install` 可以只看不装）。

板子用的就是 `install_charuco_params` 已经认定的那块（11×8、DICT_5X5_100、checker 34.31 mm），
从 `tiptop.scripts.calibrate_wrist_cam` 直接 import，全 rig 只有一处板子定义。

### 5.2 Renderer overlay gate（硬门槛，`gwm_arm/overlay_gate.py`）

sim 里这一关（GI-2）是拿 FK 对 `body_pos_w` 对到 0.0 mm 放行的；真机没有这个 oracle，只能对像素。用 `real_data_train.renderer.edge_gate` 的两个数：

- `lift`：渲染轮廓与**同朝向**强边缘的重合度减去随机基线（对比度无关）；
- `margin`：`lift` 减去把相机故意转 8° 之后同一渲染的 `lift`。**这个数才是抓错标定的**——杂乱场景给任何轮廓都有不错的 lift，只有正确位姿能赢过自己的扰动版。

```bash
$P -m gwm_hardware.gwm_arm.capture live --external-only --out-dir $D   # 顺便把 q 存进 h5
$P -m gwm_hardware.gwm_arm.overlay_gate --external-h5 $D/external_obs.h5 --out-dir $D/overlay
```

不过关按顺序查：外参（§5.1）→ 法兰立柱（`build_2f140 --flange-offset`，本 rig 假设 0）→
是不是真的在渲 2F-140。

**渲染模型注意**：SAPIEN 的 URDF loader 要求每个连杆惯性张量正定，而 cuTAMP 的
`panda_robotiq_2f_85.urdf`（我们的手臂那一半逐字复用它）里 `panda_link3` 的特征值是
`[-0.0038, +0.0033, +0.0217]`——负的。cuRobo 不看惯性所以规划模型没事，SAPIEN 直接拒绝加载。
`gwm_arm/render_model.py` 生成一个**只用于渲染**的副本，把这类张量修成正定并把改动写进
文件头注释；几何、关节、限位、mimic、mesh 路径原样复制。已实测：Panda + 2F-140 渲染正常，
张开/闭合两帧手指可见变化。

### 5.3 执行（`gwm_arm/execute.py`）

winner 是 `serialize_plan` 格式的 JSON，`tiptop.execute_plan.execute_cutamp_plan` 吃不了
（它要的是 cuRobo 的活对象）。`execute.py` 用**完全相同的 controller 调用**跑序列化格式，
免得 A/B 被「两条 arm 驱动机器人的方式不同」污染。它额外做离线计划才需要的检查：

- 计划里每个 waypoint 都是绝对位置，第一个就是采集时的 capture 位。**当前构型和计划
  `q_init` 每个关节差超过 0.02 rad 就拒绝执行**，`--go-to-start` 才会先规划一段过去；
- `--execute` 之外还要在终端确认一次；
- 结束时夹爪保持闭合（pick 计划末尾正拿着东西），`--open-after` 才松开。

一键跑完整条链：

```bash
./droid/gwm_hardware/gwm_arm/services.sh start gwm     # 共用栈 + gwm-server
$P -m gwm_hardware.gwm_arm.run_real --run-dir $D \
    --instruction "pick up the blue cup" --execute --go-to-start
```

`run_real.py` 把六个阶段各起一个进程：**进程边界就是 G-8 的显存顺序化**——planner 栈
（6–10 GB）和 gwm-server（~20 GB）不会同时占卡，而且任何一个阶段都能拿着上一阶段落盘的
产物单独重跑，这才是调试时真正会做的事。

---

## 6. 调试窗口：中间产物看什么

baseline tiptop 会开一个 Rerun 窗口。GWM arm 保留它，并加上这条 arm 特有的、否则只是
JSON 里一个数字的东西。

```bash
$P -m gwm_hardware.gwm_arm.viz_debug \
    --proposals-dir $D/proposals --h5-path $D/wrist_obs.h5 \
    --external-h5 $D/external_obs.h5 --tag <tag> --rerun
```

两个产出：

**`score_overlay_<tag>.png`** —— 先看这张。相机图上画出**每一条还在场景里的候选轨迹**，
**按它对指令拿到的 GWM 分数着色**；图例里列出每个候选的分数、M2T2 confidence、gate 判定，
选中的物体和 winner 单独标出来（winner 加粗、抓取点套白圈）。

三个刻意的设计：

1. **颜色是相对的。** GWM 的 cosine 分数在一组候选里通常只散开 ~0.01（G-28），绝对色标会
   把 16 个候选画成同一个颜色。色带拉伸到本组的 min..max，并把真实区间印在图例上——所以
   「散得开」和「几乎没散开」在图上长得不一样。**量级看数字，不要看颜色。**
2. **只画到夹爪闭合为止**，retract 段扔掉。tiptop 的 pick 计划是 `MoveFree → Pick`，而 Pick
   末尾会退回原位，所以每个候选的最后一个 waypoint 是同一个 retract 位姿。第一版把路径末端
   当成「抓取点」，16 个十字全叠在画面外的同一个像素上——这个错误本身就说明了为什么要画出来看。
3. **前段细、后段粗。** 16 条路径的前三分之二都是同一段转移，只会糊住场景；区分候选的是
   approach。

**Rerun 3D** —— 几何问题看这个：世界点云、每个簇一种颜色、所有候选轨迹（同一套配色）、
winner 加粗，外加一个 `selection` 文本面板列出两段式选择的全过程（物体级聚合排名 → 物体内
M2T2 confidence → gate）。

没有 `scores_*.json` 时（还没打分），自动退化成按 M2T2 confidence 着色，图例里会说明。

---

## 6a. 显存预算(2026-08-19 实测,32 GB 卡)

第一次真机全链路在 `gate` 阶段 CUDA OOM。逐进程量出来:

| 服务 | 显存 | 谁需要它 |
|---|---|---|
| FoundationStereo | **9342 MiB** | 只有 `capture`(腕相机深度) |
| gwm-server | ~19100 MiB | 只有 `score` |
| M2T2 | 1180 MiB | 只有 `propose` |
| cuRobo | ~740 MiB | `propose` / `gate` / `viz` |

四个全在 = 30.4 GB / 32.6 GB,cuRobo 挤不进去。

**不是 dtype 的问题**:Qwen3-VL-Embedding-8B 本来就是 `torch_dtype=torch.bfloat16`,
8B × 2 B ≈ 16 GB,是 scorer 那 19 GB 的主体,符合预期。GWM 头部是 `model.float()`(fp32),
换 bf16 能省约 0.7 GB —— **没换**:那是所有 sim 结果走过的数值路径,为 0.7 GB 冒重现性的险不划算。

**第一版修法是错的**:我先去杀 scorer。它要重新加载 16 GB 的 Qwen 权重、约 60 s,而且**每条指令都要用**。
FoundationStereo 更大、30 s 就能起、**每个场景只用一次**。改成 capture 之后释放深度服务,
实测 scorer 和 cuRobo 可以共存,峰值 **21.9 GB**,还剩 10 GB 余量 —— 整条链跑完不用再拆任何东西。

`run_real.py` 现在:capture 前确保 FoundationStereo 在(不在就起),capture 后释放它
(`--keep-depth` 可保留)。scorer 全程常驻。

## 7. 环境

| | |
|---|---|
| gwm venv | `/home/quanyi/gwm-wiser/.venv`，Python 3.11.16，torch **2.10.0+cu128**（`sm_120` 原生），transformers 4.57.6，sapien/mani_skill，lerobot |
| Qwen 权重 | `Qwen/Qwen3-VL-Embedding-8B`，HF 缓存 |
| checkpoint | `/home/quanyi/0810_gwm/checkpoint.pt`（run-1 step 34000，G-19…G-32 全部 sim 结果所用的那个） |
| gwm-server | `:8901`，`services.sh start [dummy\|gwm]` |

`.venv` 和 tiptop 的 pixi env **绝对不能合并**（D9：`transformers==4.57.6` 与 tiptop 的
py3.12 栈冲突，进程边界是设计的一部分）。gwm-server 从仓库根以 `-m droid.server.gwm_server`
启动，`real_data_train` / `gwm_wiser` 由 cwd 解析，不需要 editable install。

关服务用端口，不要 kill `$!`（G-25）：`fuser -k 8901/tcp`。
