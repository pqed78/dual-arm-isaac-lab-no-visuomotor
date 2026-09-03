# Dual Arm Handover Environment (Isaac Lab)

## Overview
This repository contains a customized reinforcement learning (RL) environment built on [Isaac Lab](https://github.com/isaac-sim/IsaacLab). The project simulates a **Dual Arm Handover Task** using two Franka robot arms. 
The goal of this environment is to train an RL policy to complete the following sequence:
1. **Pick:** The right arm picks up a green cylinder from the table.
2. **Handover:** The right arm moves the object to a designated handover zone in mid-air, while the left arm reaches for it.
3. **Place:** The left arm takes the object and successfully places it onto a red target marker.

*Note: This repository (`dual-arm-isaac-lab-no-visuomotor`) provides a **state-based** observation environment (no cameras/visuomotor included). A vision-based version using camera sensors is planned for future updates.*

## Features
- **Custom Scene Configuration (`DualArmSceneCfg`)**: Instantiates a ground plane, two Franka arms, a green cylinder object, and a red target marker.
- **Comprehensive Observations**: The policy observation space now includes full 7D poses (Position + Quaternion) for both TCPs (`pick_tcp_quat`, `place_tcp_quat`). This allows the agent to correctly orient its wrist to match the randomly rotated cube during curriculum learning.
- **Custom Reward Functions**: Dense reward shaping to guide the agents through the complex handover task. The reward system is meticulously designed to prevent local minima and RL exploits:
  - `action_penalty` (-0.01) & `action_rate_penalty` (-0.05): Penalizes large and sudden actions to induce smooth movement.
  - `tcp_floor_penalty` (-5.0): Penalty for the TCP dropping below the floor (0.0).
  - `pick_grasp_pose` (10.0): Forces a strict top-down vertical grasp when the object is on the table. However, once the object is lifted off the ground (Z > 5cm), this constraint is completely released (auto-max reward), allowing the robot to comfortably tilt its wrist to avoid kinematic singularities during horizontal transport.
  - `pick_reach` (20.0): Employs a **Dynamic Target** to enforce a strict "Elevator Drop" trajectory (hovers 12cm above the cube until XY is aligned, then drops vertically).
  - `gripper_close` (20.0): Reward for closing fingers around the cube.
  - `premature_gripper_close` (-0.5): Mildly penalizes the robot for closing its gripper when the object is far away.
  - `pick_lift` (100.0): Massive jackpot reward for lifting the object off the ground. Object mass was lowered to 0.2kg to ensure stable lifting.
  - `handover_approach` (200.0): Reward for bringing the object to the handover position. Handover height was lowered to 20cm to circumvent the Right Arm's reach limits. Also features an extremely forgiving 'is_held' margin (20cm decay) to cure the robot's fear of slipping the object during fast movements.
  - `place_reach` (100.0): Reward for the left arm reaching the handover position. A critical regex typo (`panda_hand` -> `panda_hand$`) that caused the Right Arm to steal all Left Arm rewards was fixed, finally awakening the Left Arm.
  - `place_grasp_pose` (10.0): Dense reward for the left arm to approach the cube horizontally.
  - `place_gripper_close` (50.0): Reward for the left arm closing its gripper during handover.
  - `pick_release` (100.0): Rewards the right arm for opening its gripper once the left arm has secured the object.
  - `place_to_target` (200.0): Dense distance reward for moving the object from the handover zone to the target.
  - `place_object` (1000.0): Final jackpot reward for successfully placing the object.
- **Physics Calibration & Early Termination**: Object mass optimized to 0.2kg with high friction (2.0). The episode terminates early if the object falls off the table (`Z < -0.1`).
- **Curriculum Learning**: Progress-based randomized spawning (`curriculum_events.py`) to transition the robot from a fixed tutorial state to full 360-degree rotational generalization. When resuming from a checkpoint (`--checkpoint`), the curriculum automatically skips the tutorial phase and instantly applies maximum randomization (`progress=1.0`).
- **High-Capacity Neural Network**: SKRL PPO layers expanded to `[1024, 512, 256]` to memorize the complex kinematics of dual-arm multi-stage handover.
- **Scalable RL Setup**: Configured to run thousands of environments in parallel (e.g., `num_envs=4096`).

## Installation

1. Install [Isaac Lab](https://isaac-sim.github.io/IsaacLab/) by following their official installation guide.
2. Clone this repository outside of the core Isaac Lab directory.
3. Install this project in editable mode using the Python interpreter provided by Isaac Lab:
```bash
python -m pip install -e source/dual_arm0
```

## Usage

You can verify the installation and run the environment with the provided dummy agents:

- **List available tasks:**
```bash
python scripts/list_envs.py
```

- **Run with a Random Agent:**
```bash
python scripts/random_agent.py --task=Isaac-Dual-Arm-v0
```

- **Run with a Zero-Action Agent:**
```bash
python scripts/zero_agent.py --task=Isaac-Dual-Arm-v0
```

## Training

To train the RL policy using SKRL, run the following command:

```bash
python scripts/skrl/train.py --task=Isaac-Dual-Arm-v0 --num_envs=4096
```

**Common Arguments:**
- `--num_envs=4096`: Specifies the number of environments to simulate in parallel. Adjust this based on your GPU memory.
- `--checkpoint=/path/to/model.pt`: Path to a model checkpoint to resume training from.
- `--headless`: Run the simulation without rendering the GUI (faster training).
- `--video`: Record videos during training.
- `--max_iterations=1000`: Set the maximum number of training iterations.
- `--seed=42`: Set a random seed for reproducibility.

### Tracking Progress with TensorBoard

Training logs and model checkpoints are automatically saved in the `logs/` directory. You can monitor the training progress (e.g., rewards, episode length, losses) using TensorBoard.

```bash
tensorboard --logdir=logs
```
Then, open your web browser and navigate to `http://localhost:6006`.

## Evaluation

To play/evaluate a trained model, run the following command:

```bash
python scripts/skrl/play.py --task=Isaac-Dual-Arm-v0 --num_envs=64 --checkpoint=/path/to/model.pt
```

**Common Arguments:**
- `--num_envs=64`: Use a smaller number of environments for visualization.
- `--checkpoint=/path/to/model.pt`: Path to the trained model checkpoint to evaluate.
- `--video`: Record videos of the evaluation.

---

# 듀얼 암 핸드오버 환경 (Isaac Lab)

## 개요
이 리포지토리는 [Isaac Lab](https://github.com/isaac-sim/IsaacLab) 기반으로 제작된 맞춤형 강화학습(RL) 환경을 포함하고 있습니다. 두 대의 프랭카(Franka) 로봇 팔을 이용해 **물건 건네주기(Handover) 작업**을 수행하는 시뮬레이션 환경입니다.
이 환경의 목표는 다음 과정을 완수하는 RL 정책을 학습시키는 것입니다:
1. **집기 (Pick):** 오른쪽 팔이 바닥에 있는 초록색 원기둥을 집어 올립니다.
2. **건네주기 (Handover):** 오른쪽 팔이 공중의 핸드오버 지점(Handover zone)으로 물체를 가져가고, 왼쪽 팔이 이를 받기 위해 다가갑니다.
3. **내려놓기 (Place):** 왼쪽 팔이 물체를 건네받아 빨간색 목표 지점(Target)에 성공적으로 내려놓습니다.

*참고: 현재 이 리포지토리(`dual-arm-isaac-lab-no-visuomotor`)는 카메라가 포함되지 않은 **상태 기반(State-based)** 관측 환경을 제공합니다. 비전(Vision) 센서를 활용하는 시각 기반 모델은 향후 추가될 예정입니다.*

## 주요 기능
- **맞춤형 씬 구성 (`DualArmSceneCfg`)**: 바닥, 두 대의 프랭카 로봇 팔, 25cm 길이의 초록색 바통(Baton), 그리고 빨간색 타겟 마커를 생성합니다.
- **포괄적인 관측 공간 (Observations)**: 
  - 큐브의 절대적인 정중앙 위치 및 3D 회전 각도가 기본 제공됩니다.
  - **[핵심] 실시간 끄트머리 추적(Dynamic Tracking) 관측**: 25cm 바통의 회전 상태를 실시간으로 분해하여, 각 팔(오른팔/왼팔) 쪽에 가장 가깝게 뻗어나온 8cm 끄트머리 좌표를 계산해 뇌에 직접 주입합니다. 이를 통해 인공지능이 정중앙이 아닌 양 끝단을 찰거머리처럼 정확히 조준할 수 있습니다.
- **세분화된 보상 함수 (Dense Rewards)**: 복잡한 릴레이 핸드오버 작업을 유도하고 인공지능의 기상천외한 꼼수(Exploits)를 완벽히 차단하기 위해 다음과 같이 설계되었습니다:
  - `action_penalty` & `action_rate_penalty`: 거칠거나 떨리는 관절 움직임을 억제하여 부드러운 모션 유도.
  - `tcp_floor_penalty`: 바닥 충돌 방지 페널티.
  - `pick_reach` (20.0) / `place_reach` (100.0): 각 팔이 바통 정중앙이 아닌 **각자의 방향(Y축)으로 8cm 튀어나온 끄트머리**를 향해 돌진하도록 유도합니다. 두 팔 사이에 16cm의 안전 여유 공간을 확보해 충돌을 방지합니다.
  - `gripper_close` (20.0) / `place_gripper_close` (50.0): 목표 끄트머리 반경 6cm 이내에 진입했을 때만 엄격하게 주먹을 쥐는 보상을 줍니다.
  - **[강화]** `premature_gripper_close` & `place_premature_gripper_close` (-50.0): 허공에서 꼼수로 주먹을 쥐고 대기하며 치고 다니는 현상(Fist-bumping & Fake Handshake)을 완전히 뿌리 뽑기 위해 -50.0점의 강력한 감점 폭탄을 투하합니다. 높이 제한(is_above_top) 버그를 삭제하여 수직으로 세워진 바통 끄트머리를 잡아도 합법으로 인정합니다.
  - `pick_lift` (100.0): 물체를 바닥에서 들어 올렸을 때 부여되는 잭팟 보상.
  - **[강화]** `handover_approach` (200.0): 오른팔이 끄트머리를 잡고 중앙으로 배달할 때 주어지는 잭팟 보상. 이때 **바통의 길게 남은 몸통(+Y 방향)이 완벽하게 왼팔 쪽을 향해 뻗어나오도록 손목 각도를 강제하는 방향 보너스(Pointing Bonus)**가 포함되어 있습니다.
  - `pick_release` (100.0): 왼팔이 잡은 것을 확인한 뒤 오른팔이 그립을 열어 양보했을 때 주어지는 '완벽한 릴레이' 보상.
  - `place_to_target` (200.0): 왼팔이 큐브를 들고 바닥의 빨간 원(Target)을 향해 하강할 때 주어지는 초밀집(Dense) 보상.
  - `place_object` (1000.0): 최종 안착 시 터지는 무한 우상향 잭팟 보상.
- **물리 엔진 최적화 & 조기 종료(Early Termination)**: 바통 무게를 최적화하고(0.2kg), 책상 아래(`Z < -0.1`)로 떨어지면 에피소드를 즉시 리셋합니다.
- **커리큘럼 학습 (Curriculum Learning)**: `curriculum_events.py`를 통해 바통의 스폰 난이도를 점진적으로 올리며, 체크포인트 이어서 학습 시 즉시 최고 난이도로 스폰됩니다.
- **대용량 신경망 (High-Capacity Neural Network)**: 듀얼 암의 콤보 동작을 기억할 수 있도록 SKRL PPO 모델 용량을 `[1024, 512, 256]`으로 대폭 확장.

## 설치 방법

1. [Isaac Lab](https://isaac-sim.github.io/IsaacLab/) 공식 설치 가이드를 따라 Isaac Lab을 설치합니다.
2. Isaac Lab 코어 디렉토리 외부의 경로에 이 리포지토리를 클론(Clone)합니다.
3. Isaac Lab이 설치된 Python 인터프리터를 사용하여 이 프로젝트를 편집 가능 모드(editable mode)로 설치합니다:
```bash
python -m pip install -e source/dual_arm0
```

## 사용 방법

제공되는 더미 에이전트 스크립트를 사용하여 설치를 확인하고 환경을 실행해 볼 수 있습니다:

- **사용 가능한 태스크 목록 확인:**
```bash
python scripts/list_envs.py
```

- **랜덤 에이전트로 실행:**
```bash
python scripts/random_agent.py --task=Isaac-Dual-Arm-v0
```

- **Zero-Action 에이전트로 실행:**
```bash
python scripts/zero_agent.py --task=Isaac-Dual-Arm-v0
```

## 훈련 방법 (Training)

SKRL 라이브러리를 사용하여 강화학습 모델을 훈련하려면 아래 명령어를 실행하세요:

```bash
python scripts/skrl/train.py --task=Isaac-Dual-Arm-v0 --num_envs 4096
```

**주요 인자 (Arguments):**
- `--num_envs`: 동시에 시뮬레이션할 환경의 개수를 지정합니다. GPU 메모리에 맞게 조절하세요.
- `--checkpoint=/경로/model.pt`: 이전에 저장된 모델 가중치 파일(체크포인트)을 불러와서 이어서 훈련할 때 사용합니다.
- `--headless`: 화면 렌더링 없이 백그라운드에서 실행합니다 (훈련 속도 향상).
- `--video`: 훈련 중 시뮬레이션 영상을 녹화합니다.
- `--max_iterations=1000`: 최대 훈련 반복(iteration) 횟수를 지정합니다.
- `--seed=42`: 실험 재현성을 위해 난수 시드(seed)를 고정합니다.

### 텐서보드(TensorBoard)로 학습 진행 상황 확인하기

학습 로그와 모델 체크포인트는 자동으로 `logs/` 폴더에 저장됩니다. 텐서보드를 사용하면 훈련 진행 상황(보상, 에피소드 길이, 손실 등)을 시각적으로 모니터링할 수 있습니다.

```bash
tensorboard --logdir=logs
```
명령어를 실행한 후 웹 브라우저에서 `http://localhost:6006`에 접속하세요.

## 평가 방법 (Evaluation)

학습이 완료된 모델을 불러와서 시뮬레이션 환경에서 테스트(평가)하려면 아래 명령어를 실행하세요:

```bash
python scripts/skrl/play.py --task=Isaac-Dual-Arm-v0 --num_envs=64 --checkpoint=/경로/model.pt
```

**주요 인자 (Arguments):**
- `--num_envs=64`: 시각적 확인을 위해 훈련 때보다 적은 수의 환경을 실행합니다.
- `--checkpoint=/경로/model.pt`: 평가할 훈련된 모델의 체크포인트 경로를 지정해야 합니다.
- `--video`: 평가 과정을 영상으로 녹화합니다.