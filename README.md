# XLeRobot 拖动示教录制器
> **测试环境：WSL2 (Ubuntu 22.04) + Windows 11**
> 
为 [LeRobot](https://github.com/huggingface/lerobot) / SO-101 设计的零成本拖动示教数据采集方案。

**无需主臂，无需 VR，无需手柄，用手拖就行**

---

## 为什么要做这个

LeRobot 官方的遥操作需要主臂、VR 头盔或者游戏手柄。这个项目让你**直接用手拖动从臂**录制专家轨迹，回放时自动生成 LeRobot 标准数据集。

---





## ⚠️ 第零步：把机器人固定好

没有底盘的xlerobot很轻，拖动的时候**它会跑**。建议使用专业级固定装置。

<img width="1706" height="1279" alt="pinned" src="https://github.com/user-attachments/assets/66269de7-3b97-47a3-8313-1c31c3b37fad" />

*图 1：专业级固定装置。*

---

## 第一步：连接并校准（一次性）

如果你还没校准过机械臂，先跑官方命令：

```bash
# 左臂
lerobot-calibrate --robot.type=so101_follower --robot.port=/dev/ttyACM0 --robot.id=my_left_arm

# 右臂
lerobot-calibrate --robot.type=so101_follower --robot.port=/dev/ttyACM1 --robot.id=my_right_arm
```

> **WSL2 用户注意**：需要先用 `usbipd` 把 USB 设备 attach 到 WSL2。不会的话搜一下 `usbipd wsl2`，有很多教程，或者问问ai。

---

## 第二步：启动脚本

```bash
python xlerobot_kinesthetic_recorder.py
```

浏览器打开：`http://localhost:7863`

你应该看到这个界面：

![控制面板](https://user-images.githubusercontent.com/你的ID/ui.png)

*图 2：控制面板。按钮一看就懂，看不懂就随便点，点到有反应为止。*

---

## 第三步：拖动示教

1. **勾选要示教的手臂**（左臂 / 右臂 / 双臂）
2. 点 **🔌 掉电** —— 舵机掉电，手臂变软可以拖动
3. **用手拖动机械臂** 完成你的任务
4. 点 **⏹️ 停止示教** —— 轨迹已保存在内存中

> **建议**：示教结束时尽量把机械臂拖回零位附近，这样回放更顺滑。

---

## 第四步：复位（推荐）

点 **🔄 复位**，双臂回到零位。

这是为了让回放从标准姿势开始。如果你示教结束时已经在零位附近，这一步可以跳过。

---

## 第五步：回放并自动生成数据集

1. 确保摄像头画面干净，别挡着
2. 点 **🎬 回放并录制视频**
3. 机械臂自动执行刚才的轨迹
4. **LeRobot 数据集自动生成**在：
   ```
   /mnt/g/lerobot/datasets/so101_task_v1/
   ├── data/chunk-000/episode_000000.parquet
   ├── videos/chunk-000/episode_000000_camera1.mp4
   ├── episodes.jsonl
   ├── info.json
   └── modality.json
   ```

---

## 第六步：训练

```bash
lerobot-train \
  --dataset.repo_id=local/so101_task_v1 \
  --dataset.root=/mnt/g/lerobot/datasets/so101_task_v1 \
  --dataset.video_backend=pyav
```

---

## 按钮说明

| 按钮 | 作用 | 什么时候点 |
|------|------|-----------|
| 🔌 掉电 | 舵机掉电，手臂可以手拖 | 拖动示教前 |
| ⚡ 上电 | 锁定当前位置 | 摆好姿势后 |
| 🤏 开始示教 | 开始记录关节角度 | 准备拖动时 |
| ⏹️ 停止示教 | 停止记录 | 拖动完成后 |
| 🔄 复位 | 双臂回到零位 | 回放前（可选） |
| 🎬 回放并录制 | 执行轨迹 + 自动生成数据集 | 复位后 |
| 🗑️ 清除轨迹 | 丢弃当前记录 | 录砸了重来 |
| 💾 保存轨迹 | 把 JSON 轨迹存到本地 | 想留备份时 |
| 📂 加载轨迹 | 读取之前保存的 JSON | 复用旧轨迹 |

---

## 硬件要求

- XLeRobot SO-101（只需从臂，无需主臂）
- 1-3 个 USB 摄像头（支持头部 / 左手腕 / 右手腕视角）
- 2 个 USB 口接机械臂
- 胶带（必需）

---

## 环境要求

- Ubuntu / WSL2
- Python 3.10+
- LeRobot v0.6.0+
- 依赖：`pip install lerobot pandas pyarrow gradio opencv-python`

---

## 数据集格式

回放时自动导出 LeRobot v2.1 标准格式，可直接用于：
- Diffusion Policy
- ACT (Action Chunking with Transformers)
- SmolVLA

---

## 许可证

MIT

