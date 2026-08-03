
# XLeRobot Kinesthetic Recorder


Zero-arm teleoperation data collection for [LeRobot](https://github.com/huggingface/lerobot) / SO-101.

**No leader arm. No VR. No gamepad. Just drag.**

<img width="426" height="240" alt="HnVideoEditor_2026_08_03_162920530" src="https://github.com/user-attachments/assets/99780aa7-19f4-43b8-9071-539191576f36" />




[中文说明见下方](#中文说明)

## Why

Official LeRobot teleoperation requires a leader arm, VR headset, or gamepad. 
This project lets you **drag the follower arm directly by hand** to record expert 
trajectories, then replay and automatically export to LeRobot dataset format.


## Hardware

- XLeRobot SO-101 (follower arm only, no leader arm needed)
- 1-3x USB cameras (head, left_wrist, right_wrist supported)
- ## ⚠️ Real-World Tips

### Step 0: Secure Your Robot

SO-101 is lightweight. When you drag it by hand, **it will move**.
We recommend professional-grade stabilization equipment.

![Professional Stabilization Equipment](https://user-images.githubusercontent.com/你的ID/tape.png)

*Figure 1: Advanced anti-slip system (duct tape).*

### Step 1: Open the UI

<img width="426" height="240" alt="HnVideoEditor_2026_08_03_162920530" src="https://github.com/user-attachments/assets/647cb424-260c-4d21-bff9-57db4b986417" />
<img width="945" height="1269" alt="ui" src="https://github.com/user-attachments/assets/1a2afcd6-09b5-4911-9d9c-705c5d9c2143" />


*Figure 2: The control panel. Buttons are self-explanatory. If not, click randomly until something moves.*



![Teaching Demo](assets/demo.gif)  <!-- 你的GIF放这里 -->

## Quick Start

```bash
# Install dependencies
pip install lerobot pandas pyarrow gradio opencv-python

# Calibrate arms first (official LeRobot command)
lerobot-calibrate --robot.type=so101_follower --robot.port=/dev/ttyACM0 --robot.id=my_left_arm
lerobot-calibrate --robot.type=so101_follower --robot.port=/dev/ttyACM1 --robot.id=my_right_arm

# Run
python xlerobot_kinesthetic_recorder.py
# Open browser: http://localhost:7863
```

## Dataset Output

Replay automatically generates LeRobot-compatible dataset at:
```
/mnt/g/lerobot/datasets/so101_task_v1/
├── data/chunk-000/episode_000000.parquet
├── videos/chunk-000/episode_000000_camera1.mp4
├── episodes.jsonl
├── info.json
└── modality.json
```

Train directly with:
```bash
lerobot-train \
  --dataset.repo_id=local/so101_task_v1 \
  --dataset.root=/path/to/datasets/so101_task_v1 \
  --dataset.video_backend=pyav
```

## Requirements

- Ubuntu / WSL2
- Python 3.10+
- LeRobot v0.6.0+
- 2x USB ports for SO-101 arms
- 1-3x USB cameras

## License

MIT

---

## 中文说明

本项目为 XLeRobot SO-101 机械臂提供零成本拖动示教数据采集方案。

**无需主臂、无需 VR、无需手柄，直接用手拖动从臂即可录制专家轨迹。**

回放时自动生成 LeRobot v2.1 标准数据集，可直接用于 Diffusion Policy / ACT / SmolVLA 等模仿学习模型训练。

核心特点：
- 掉电拖动示教（kinesthetic teaching）
- 双臂数据同步录制
- 三路摄像头同步录制（head / left_wrist / right_wrist）
- 自动导出 LeRobot 标准格式
```

