import time
import os
import json
import cv2
import numpy as np
import gradio as gr
import threading
import shutil
from lerobot.robots.so_follower.so_follower import SOFollower
from lerobot.robots.so_follower.config_so_follower import SOFollowerRobotConfig

# ========== 配置 ==========
SAVE_DIR = "/mnt/g/lerobot/videos"
TRAJ_DIR = "/mnt/g/lerobot/trajectories"
os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(TRAJ_DIR, exist_ok=True)
EPISODE_LEN = 500

JOINT_NAMES = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
CAMERAS = {"head": 4, "left_wrist": 2, "right_wrist": 0}

# ========== LeRobot 数据集配置（用户按需修改） ==========
# 依赖：pip install pandas pyarrow
LEROBOT_DATASET_DIR = "/mnt/g/lerobot/datasets/so101_task_v1"   # 数据集输出目录
LEROBOT_TASK_NAME = "put object in place"                         # 任务描述，训练时会用到
LEROBOT_FPS = 20                                                  # 与回放频率一致（0.05s = 20Hz）
# 相机名称映射到 LeRobot 标准命名（回放目前只录 head，扩展多相机需改 replay_loop）
LEROBOT_CAMERA_MAP = {
    "head": "camera1",
    "left_wrist": "camera2",
    "right_wrist": "camera3"
}

# ========== 全局状态 ==========
class RobotState:
    def __init__(self):
        self.left_pos = [0.0] * 6
        self.right_pos = [0.0] * 6
        self.teaching = False
        self.teach_left = True
        self.teach_right = False
        self.replaying = False
        self.frame_count = 0
        self.trajectory = []
        self.status = "就绪（未连接）"
        self.running = True
        self.lock = threading.Lock()
        self.connected = False
        self.video_writer = None
        self.video_path = ""
        self.current_traj_name = ""

state = RobotState()

# ========== 连接硬件 ==========
def safe_connect():
    global left, right, caps
    try:
        print("连接左手...")
        left = SOFollower(SOFollowerRobotConfig(id="my_left_arm", port="/dev/ttyACM0", use_degrees=True))
        left.connect()
        print("连接右手...")
        right = SOFollower(SOFollowerRobotConfig(id="my_right_arm", port="/dev/ttyACM1", use_degrees=True))
        right.connect()
        print("连接摄像头...")
        caps = {}
        for name, idx in CAMERAS.items():
            cap = cv2.VideoCapture(idx)
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            caps[name] = cap
        state.connected = True
        print("✅ 全部连接成功")
        return True
    except Exception as e:
        state.status = f"连接失败: {e}"
        print(f"❌ {e}")
        return False

safe_connect()

def read_positions():
    if not state.connected:
        return
    try:
        l_obs = left.get_observation()
        r_obs = right.get_observation()
        with state.lock:
            for i, name in enumerate(JOINT_NAMES):
                state.left_pos[i] = l_obs.get(f"{name}.pos", 0.0)
                state.right_pos[i] = r_obs.get(f"{name}.pos", 0.0)
    except:
        pass

read_positions()

# ========== LeRobot 格式辅助函数 ==========

def _get_next_episode_index():
    """读取已有 episodes，返回下一个编号"""
    episodes_path = os.path.join(LEROBOT_DATASET_DIR, "episodes.jsonl")
    if not os.path.exists(episodes_path):
        return 0
    max_idx = -1
    with open(episodes_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                try:
                    ep = json.loads(line)
                    max_idx = max(max_idx, ep.get("episode_index", -1))
                except:
                    pass
    return max_idx + 1

def _update_lerobot_metadata(episode_idx, num_frames, task_name):
    """更新 LeRobot 数据集的 metadata 文件（episodes/tasks/info/modality）"""
    os.makedirs(LEROBOT_DATASET_DIR, exist_ok=True)

    # episodes.jsonl — 追加
    episodes_path = os.path.join(LEROBOT_DATASET_DIR, "episodes.jsonl")
    with open(episodes_path, "a", encoding="utf-8") as f:
        f.write(json.dumps({"episode_index": episode_idx, "num_frames": num_frames}, ensure_ascii=False) + "\n")

    # tasks.jsonl — 去重追加
    tasks_path = os.path.join(LEROBOT_DATASET_DIR, "tasks.jsonl")
    existing_tasks = {}
    if os.path.exists(tasks_path):
        with open(tasks_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        t = json.loads(line)
                        existing_tasks[t.get("task", "")] = t.get("task_index", 0)
                    except:
                        pass

    if task_name not in existing_tasks:
        new_idx = len(existing_tasks)
        with open(tasks_path, "a", encoding="utf-8") as f:
            f.write(json.dumps({"task_index": new_idx, "task": task_name}, ensure_ascii=False) + "\n")

    # info.json — 重新统计
    all_episodes = []
    if os.path.exists(episodes_path):
        with open(episodes_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        all_episodes.append(json.loads(line))
                    except:
                        pass

    total_episodes = len(all_episodes)
    total_frames = sum(ep.get("num_frames", 0) for ep in all_episodes)

    info = {
        "codebase_version": "2.1.0",
        "dataset_type": "video",
        "fps": LEROBOT_FPS,
        "num_episodes": total_episodes,
        "num_frames": total_frames,
        "splits": {"train": f"0:{total_episodes}"}
    }
    with open(os.path.join(LEROBOT_DATASET_DIR, "info.json"), "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2, ensure_ascii=False)

    # modality.json — 首次创建，定义数据模态
    modality_path = os.path.join(LEROBOT_DATASET_DIR, "modality.json")
    if not os.path.exists(modality_path):
        total_dim = len(JOINT_NAMES) * 2  # 双臂 = 左6 + 右6，按你的硬件实际修改
        modality = {
            "features": {
                "action": {
                    "dtype": "float32",
                    "shape": [total_dim],
                    "names": None
                },
                "observation.state": {
                    "dtype": "float32",
                    "shape": [total_dim],
                    "names": None
                },
                "timestamp": {
                    "dtype": "float32",
                    "shape": [1],
                    "names": None
                }
            }
        }
        # 添加已映射的相机模态（目前回放只录制 head，扩展时在此添加）
        for cam_name, lerobot_name in LEROBOT_CAMERA_MAP.items():
            modality["features"][f"observation.images.{lerobot_name}"] = {
                "dtype": "video",
                "shape": [480, 640, 3],
                "names": None
            }
        with open(modality_path, "w", encoding="utf-8") as f:
            json.dump(modality, f, indent=2, ensure_ascii=False)

def save_lerobot_episode(records, video_paths, episode_idx, task_name):
    """将单次回放数据保存为 LeRobot v2.1 格式
    records: List[Dict] — 每帧的 action/state/timestamp 等
    video_paths: Dict[str, str] — {camera_name: video_file_path}
    """
    import pandas as pd

    # 1. 写 Parquet
    data_dir = os.path.join(LEROBOT_DATASET_DIR, "data", "chunk-000")
    os.makedirs(data_dir, exist_ok=True)
    df = pd.DataFrame(records)
    parquet_path = os.path.join(data_dir, f"episode_{episode_idx:06d}.parquet")
    df.to_parquet(parquet_path, index=False, engine="pyarrow")

    # 2. 复制视频到标准路径
    video_dir = os.path.join(LEROBOT_DATASET_DIR, "videos", "chunk-000")
    os.makedirs(video_dir, exist_ok=True)
    for cam_name, src_path in video_paths.items():
        if src_path and os.path.exists(src_path):
            lerobot_cam_name = LEROBOT_CAMERA_MAP.get(cam_name, cam_name)
            dst_path = os.path.join(video_dir, f"episode_{episode_idx:06d}_{lerobot_cam_name}.mp4")
            shutil.copy2(src_path, dst_path)

    # 3. 更新 metadata
    _update_lerobot_metadata(episode_idx, len(records), task_name)

    return LEROBOT_DATASET_DIR

# ========== 示教记录线程 ==========
def teaching_loop():
    while state.running:
        if not state.connected or not state.teaching:
            time.sleep(0.05)
            continue
        try:
            l_obs = left.get_observation()
            r_obs = right.get_observation()
            l_pos = [l_obs.get(f"{k}.pos", 0.0) for k in JOINT_NAMES]
            r_pos = [r_obs.get(f"{k}.pos", 0.0) for k in JOINT_NAMES]
            with state.lock:
                state.trajectory.append({"left": l_pos, "right": r_pos})
                state.frame_count = len(state.trajectory)
            state.status = f"🤏 示教记录中... {state.frame_count}帧"
        except Exception as e:
            pass
        time.sleep(0.05)

teaching_thread = threading.Thread(target=teaching_loop, daemon=True)
teaching_thread.start()

# ========== 回放录制线程 ==========
def replay_loop():
    global state
    traj = []
    with state.lock:
        traj = state.trajectory.copy()

    if not traj:
        state.status = "⚠️ 没有轨迹可回放"
        state.replaying = False
        return

    total = min(len(traj), EPISODE_LEN)
    state.status = f"🎬 回放录制中... 0/{total}"

    # 创建三路视频写入器（head + left_wrist + right_wrist）
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    writers = {}
    video_paths = {}
    for cam_name in ["head", "left_wrist", "right_wrist"]:
        path = f"{SAVE_DIR}/replay_{timestamp}_{cam_name}.mp4"
        writers[cam_name] = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*'mp4v'), 20.0, (640, 480))
        video_paths[cam_name] = path
    state.video_writer = writers["head"]   # UI 兼容

    # ========== LeRobot 数据收集（新增）==========
    lerobot_records = []
    episode_idx = _get_next_episode_index()

    for i in range(total):
        if not state.running:
            break

        frame = traj[i]
        l_act = {f"{JOINT_NAMES[j]}.pos": frame["left"][j] for j in range(6)}
        r_act = {f"{JOINT_NAMES[j]}.pos": frame["right"][j] for j in range(6)}

        try:
            left.send_action(l_act)
            right.send_action(r_act)
        except Exception as e:
            pass

        # 读取三路摄像头并分别写入视频
        for cam_name, writer in writers.items():
            try:
                ret, img = caps[cam_name].read()
                if ret:
                    writer.write(img)
            except:
                pass

        # ========== 记录当前帧到 LeRobot（新增）==========
        action_vec = frame["left"] + frame["right"]          # 双臂拼接
        state_vec = action_vec                                # 回放时执行即观测
        lerobot_records.append({
            "action": action_vec,
            "observation.state": state_vec,
            "timestamp": i * (1.0 / LEROBOT_FPS),
            "episode_index": episode_idx,
            "task_index": 0,
            "frame_index": i,
        })

        state.status = f"🎬 回放录制中... {i+1}/{total}"
        time.sleep(0.05)

    # 关闭视频
    # 关闭三路视频
    for writer in writers.values():
        writer.release()
    state.video_writer = None

    # ========== 回放结束后保存 LeRobot 格式（新增）==========
    if lerobot_records:
        try:
            save_lerobot_episode(lerobot_records, video_paths, episode_idx, LEROBOT_TASK_NAME)
            status_msg = f"✅ 视频已保存: {state.video_path} | LeRobot episode {episode_idx} 已生成 ({len(lerobot_records)}帧)"
        except Exception as e:
            status_msg = f"✅ 视频已保存: {state.video_path} | LeRobot 保存失败: {e}"
    else:
        status_msg = f"✅ 视频已保存: {state.video_path}"

    with state.lock:
        state.replaying = False
        state.status = status_msg

# ========== 轨迹文件操作 ==========
def save_trajectory(name):
    """保存当前轨迹到文件"""
    if not state.trajectory:
        return "⚠️ 没有轨迹可保存", gr.update(choices=list_saved_traj())

    if not name or name.strip() == "":
        name = f"traj_{time.strftime('%m%d_%H%M%S')}"

    filepath = os.path.join(TRAJ_DIR, f"{name}.json")
    with open(filepath, 'w') as f:
        json.dump({
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "frames": len(state.trajectory),
            "trajectory": state.trajectory
        }, f, indent=2)

    return f"💾 轨迹已保存: {name} ({len(state.trajectory)}帧)", gr.update(choices=list_saved_traj())

def load_trajectory(name):
    """从文件加载轨迹"""
    if not name:
        return "⚠️ 请选择轨迹文件", gr.update()

    filepath = os.path.join(TRAJ_DIR, f"{name}.json")
    if not os.path.exists(filepath):
        return f"❌ 文件不存在: {name}", gr.update()

    try:
        with open(filepath, 'r') as f:
            data = json.load(f)

        with state.lock:
            state.trajectory = data["trajectory"]
            state.frame_count = len(state.trajectory)

        return f"📂 已加载轨迹: {name} ({len(state.trajectory)}帧)", gr.update()
    except Exception as e:
        return f"❌ 加载失败: {e}", gr.update()

def delete_trajectory(name):
    """删除轨迹文件"""
    if not name:
        return "⚠️ 请选择要删除的轨迹", gr.update(choices=list_saved_traj())

    filepath = os.path.join(TRAJ_DIR, f"{name}.json")
    if os.path.exists(filepath):
        os.remove(filepath)
        return f"🗑️ 已删除: {name}", gr.update(choices=list_saved_traj())
    return f"❌ 文件不存在: {name}", gr.update(choices=list_saved_traj())

def list_saved_traj():
    """列出所有保存的轨迹"""
    files = []
    if os.path.exists(TRAJ_DIR):
        for f in sorted(os.listdir(TRAJ_DIR)):
            if f.endswith('.json'):
                files.append(f.replace('.json', ''))
    return files

# ========== UI 回调 ==========
def start_teaching(teach_left, teach_right):
    if not state.connected:
        return "❌ 机器人未连接"
    if state.replaying:
        return "🎬 请先等待回放结束"

    with state.lock:
        state.teaching = True
        state.teach_left = teach_left
        state.teach_right = teach_right
        state.trajectory = []
        state.frame_count = 0

    msg_parts = []
    try:
        if teach_left:
            left.bus.disable_torque()
            msg_parts.append("左臂掉电")
        else:
            msg_parts.append("左臂上电")
    except Exception as e:
        return f"左臂失败: {e}"

    try:
        if teach_right:
            right.bus.disable_torque()
            msg_parts.append("右臂掉电")
        else:
            msg_parts.append("右臂上电")
    except Exception as e:
        return f"右臂失败: {e}"

    return f"🤏 示教开始！{' + '.join(msg_parts)}。拖动机械臂完成任务并拖回零位，然后点'停止示教'"

def stop_teaching():
    if not state.teaching:
        return "未在示教模式"

    with state.lock:
        state.teaching = False

    try:
        time.sleep(0.1)
        l_obs = left.get_observation()
        r_obs = right.get_observation()
        l_act = {f"{k}.pos": l_obs.get(f"{k}.pos", 0.0) for k in JOINT_NAMES}
        r_act = {f"{k}.pos": r_obs.get(f"{k}.pos", 0.0) for k in JOINT_NAMES}
        left.send_action(l_act)
        right.send_action(r_act)
    except Exception as e:
        pass

    traj_len = len(state.trajectory)
    return f"⏹️ 示教结束，记录了 {traj_len} 帧。\n👉 下一步：点击'复位'精确回零位，然后离开画面，再点'回放并录制'"

def do_reset():
    if not state.connected:
        return "❌ 未连接"
    if state.teaching:
        return "🤏 请先停止示教"
    if state.replaying:
        return "🎬 请先等待回放结束"

    try:
        l_act = {f"{k}.pos": 0.0 for k in JOINT_NAMES}
        r_act = {f"{k}.pos": 0.0 for k in JOINT_NAMES}
        left.send_action(l_act)
        right.send_action(r_act)
        time.sleep(0.3)
        read_positions()

        if state.trajectory:
            return "✅ 已复位到零位。\n👉 下一步：离开画面，点击'回放并录制'"
        else:
            return "已复位到零位"
    except Exception as e:
        return f"复位失败: {e}"

def start_replay():
    if not state.connected:
        return "❌ 机器人未连接"
    if state.teaching:
        return "🤏 请先停止示教"
    if state.replaying:
        return "已经在回放中"
    if len(state.trajectory) == 0:
        return "⚠️ 没有轨迹，请先示教或加载"

    with state.lock:
        state.replaying = True

    replay_thread = threading.Thread(target=replay_loop, daemon=True)
    replay_thread.start()
    return "🎬 回放录制开始！机器人正在自动执行，请保持画面干净..."

def clear_trajectory():
    with state.lock:
        state.trajectory = []
        state.frame_count = 0
    return "🗑️ 轨迹已清除", gr.update(choices=list_saved_traj())

def power_off(power_left, power_right):
    if not state.connected:
        return "❌ 请先连接硬件"
    if state.teaching:
        return "⚠️ 请先停止示教"
    if state.replaying:
        return "⚠️ 请先停止回放"

    msg_parts = []
    try:
        if power_left:
            left.bus.disable_torque()
            msg_parts.append("左臂已掉电")
    except Exception as e:
        return f"左臂掉电失败: {e}"

    try:
        if power_right:
            right.bus.disable_torque()
            msg_parts.append("右臂已掉电")
    except Exception as e:
        return f"右臂掉电失败: {e}"

    if not msg_parts:
        return "⚠️ 请至少选择一只手臂"

    return f"🔌 {' + '.join(msg_parts)}。现在可以手动拖动摆姿势，摆好后点击'上电'"

def power_on(power_left, power_right):
    if not state.connected:
        return "❌ 请先连接硬件"

    msg_parts = []
    try:
        if power_left:
            l_obs = left.get_observation()
            l_act = {f"{k}.pos": l_obs.get(f"{k}.pos", 0.0) for k in JOINT_NAMES}
            left.send_action(l_act)
            msg_parts.append("左臂已上电")
    except Exception as e:
        return f"左臂上电失败: {e}"

    try:
        if power_right:
            r_obs = right.get_observation()
            r_act = {f"{k}.pos": r_obs.get(f"{k}.pos", 0.0) for k in JOINT_NAMES}
            right.send_action(r_act)
            msg_parts.append("右臂已上电")
    except Exception as e:
        return f"右臂上电失败: {e}"

    if not msg_parts:
        return "⚠️ 请至少选择一只手臂"

    return f"⚡ {' + '.join(msg_parts)}。当前位置已锁定"

def disconnect_all():
    state.running = False
    state.connected = False
    try:
        left.disconnect()
    except:
        pass
    try:
        right.disconnect()
    except:
        pass
    try:
        for cap in caps.values():
            cap.release()
    except:
        pass
    return "已断开连接"

# ========== Gradio UI ==========
def get_ui_state():
    with state.lock:
        left_vals = state.left_pos.copy()
        right_vals = state.right_pos.copy()
        status = state.status
        traj_len = len(state.trajectory)

    frames = {}
    for name, cap in caps.items() if caps is not None else []:
        try:
            ret, frame = cap.read()
            if ret:
                frames[name] = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            else:
                frames[name] = np.zeros((480, 640, 3), dtype=np.uint8)
        except:
            frames[name] = np.zeros((480, 640, 3), dtype=np.uint8)

    if not frames:
        frames = {"head": np.zeros((480, 640, 3)), "left_wrist": np.zeros((480, 640, 3)), "right_wrist": np.zeros((480, 640, 3))}

    return (
        frames["head"], frames["left_wrist"], frames["right_wrist"],
        status,
        traj_len,
        left_vals[0], left_vals[1], left_vals[2], left_vals[3], left_vals[4], left_vals[5],
        right_vals[0], right_vals[1], right_vals[2], right_vals[3], right_vals[4], right_vals[5],
    )

with gr.Blocks(title="XLeRobot Video Recorder") as demo:
    gr.Markdown("## 📹 XLeRobot 示教回放录视频 + LeRobot 数据集")
    gr.Markdown("**流程：掉电摆姿势 → 上电 → 示教（拖）→ 停止 → 保存轨迹 → 复位 → 离开画面 → 回放录制视频 & LeRobot 数据**")

    with gr.Row():
        status = gr.Textbox(value="就绪（未连接）", label="状态", interactive=False)
        traj_info = gr.Number(value=0, label="当前轨迹帧数", interactive=False)

    with gr.Row():
        with gr.Column(scale=2):
            gr.Markdown("### 摄像头")
            img_head = gr.Image(label="头部", streaming=True)
            img_lw = gr.Image(label="左手腕", streaming=True)
            img_rw = gr.Image(label="右手腕", streaming=True)

        with gr.Column(scale=1):
            gr.Markdown("### 左臂位置")
            left_sliders = []
            for name in JOINT_NAMES:
                s = gr.Slider(minimum=-180, maximum=180, value=0, step=1, label=name, interactive=False)
                left_sliders.append(s)

            gr.Markdown("### 右臂位置")
            right_sliders = []
            for name in JOINT_NAMES:
                s = gr.Slider(minimum=-180, maximum=180, value=0, step=1, label=name, interactive=False)
                right_sliders.append(s)

    with gr.Row():
        with gr.Column():
            gr.Markdown("### 示教设置")
            chk_teach_left = gr.Checkbox(label="左臂掉电示教", value=True)
            chk_teach_right = gr.Checkbox(label="右臂掉电示教", value=False)

        with gr.Column():
            gr.Markdown("### 操作")
            btn_teach = gr.Button("🤏 开始示教", variant="primary")
            btn_stop_teach = gr.Button("⏹️ 停止示教")
            btn_reset = gr.Button("🔄 复位")
            btn_replay = gr.Button("🎬 回放并录制视频", variant="stop")
            btn_clear = gr.Button("🗑️ 清除轨迹")

    with gr.Row():
        with gr.Column():
            gr.Markdown("### 电源控制（摆姿势用）")
            chk_power_left = gr.Checkbox(label="左臂", value=True)
            chk_power_right = gr.Checkbox(label="右臂", value=False)
            btn_power_off = gr.Button("🔌 掉电", variant="secondary")
            btn_power_on = gr.Button("⚡ 上电", variant="secondary")

        with gr.Column():
            gr.Markdown("### 系统")
            btn_disconnect = gr.Button("⚠️ 断开连接")

    with gr.Row():
        with gr.Column():
            gr.Markdown("### 💾 轨迹管理")
            traj_name_input = gr.Textbox(label="轨迹名称", placeholder="输入名称保存，留空用时间戳")
            btn_save_traj = gr.Button("💾 保存当前轨迹")
            saved_traj_list = gr.Dropdown(label="已保存的轨迹", choices=list_saved_traj(), interactive=True)
            btn_load_traj = gr.Button("📂 加载选中轨迹")
            btn_delete_traj = gr.Button("🗑️ 删除选中轨迹")

    all_sliders = left_sliders + right_sliders

    btn_teach.click(start_teaching, inputs=[chk_teach_left, chk_teach_right], outputs=status)
    btn_stop_teach.click(stop_teaching, outputs=status)
    btn_reset.click(do_reset, outputs=status)
    btn_replay.click(start_replay, outputs=status)
    btn_clear.click(clear_trajectory, outputs=[status, saved_traj_list])
    btn_power_off.click(power_off, inputs=[chk_power_left, chk_power_right], outputs=status)
    btn_power_on.click(power_on, inputs=[chk_power_left, chk_power_right], outputs=status)
    btn_disconnect.click(disconnect_all, outputs=status)

    # 轨迹管理按钮
    btn_save_traj.click(save_trajectory, inputs=traj_name_input, outputs=[status, saved_traj_list])
    btn_load_traj.click(load_trajectory, inputs=saved_traj_list, outputs=[status, traj_info])
    btn_delete_traj.click(delete_trajectory, inputs=saved_traj_list, outputs=[status, saved_traj_list])

    timer = gr.Timer(value=0.1, active=True)
    timer.tick(get_ui_state, outputs=[img_head, img_lw, img_rw, status, traj_info] + all_sliders)

if __name__ == "__main__":
    print("=" * 60)
    print("📹 XLeRobot 示教回放录视频 + LeRobot 数据集生成已启动")
    print("🌐 浏览器打开: http://localhost:7863")
    print("💾 轨迹保存到 G:/lerobot/trajectories/")
    print("💾 轨迹保存到 G:/lerobot/trajectories/")
    print("🤖 LeRobot 数据集输出到 G:/lerobot/datasets/so101_task_v1/")
    print("🔌 掉电摆姿势 → ⚡ 上电 → 🤏 示教 → 💾 保存 → 🎬 回放录制（同时生成 LeRobot 数据）")
    print("=" * 60)
    demo.launch(server_name="0.0.0.0", server_port=7863)