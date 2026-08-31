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
  - `pick_grasp_pose` (10.0): Global posture reward. Highly incentivizes the robot to align its wrist with the short axis of the cube. Supports 180-degree symmetric grips (`torch.abs` dot product) to prevent orientation confusion.
  - `pick_reach` (20.0): Employs a **Dynamic Target** to enforce a strict "Elevator Drop" trajectory (hovers 12cm above the cube until XY is aligned, then drops vertically). Distance margin was aggressively tightened (4cm -> 1.5cm) to prevent the "Edge Pinching" local minimum and force a secure, deep grip.
  - `gripper_close` (20.0): Reward for closing fingers around the cube. Height constraints were relaxed to allow grasping the top edges of the cube.
  - `premature_gripper_close` (-0.5): Mildly penalizes the robot for closing its gripper when the object is far away to prevent fist-bumping without freezing exploration.
  - `pick_lift` (50.0): Massive jackpot reward for lifting the object off the ground.
  - `handover_approach` (20.0): Reward for bringing the object to the center handover position. Explicitly requires the TCP to be holding the object to prevent "Batting/Flicking" exploits.
  - `place_reach` (10.0): Reward for the left arm reaching the handover position, activated only when the right arm brings it to the zone.
  - `place_grasp_pose` (10.0): Dense reward for the left arm to approach the cube horizontally and pinch the side faces, preventing collisions with the right arm holding the top.
  - `place_gripper_close` (20.0): Reward for the left arm closing its gripper around the object during handover.
  - `pick_release` (20.0): Rewards the right arm for opening its gripper once the left arm has secured the object. Crucial for breaking the "Tug-of-War" & "Statue" local minima.
  - `place_to_target` (100.0): Highly dense distance reward for moving the object from the handover zone to the target. Prevents a "Reward Valley" drop-off.
  - `place_object` (200.0): Final jackpot reward for successfully placing the object on the target. Upgraded to a "True Place" logic: explicitly requires both arms to open their grippers and move away from the object.
- **Physics Calibration & Early Termination**: High friction (2.0) applied to the cube (`RigidBodyMaterialCfg`). The episode terminates early if the object falls off the table (`Z < -0.1`), drastically improving sample efficiency.
- **Curriculum Learning**: Progress-based randomized spawning (`curriculum_events.py`) to transition the robot from a fixed tutorial state to full 360-degree rotational generalization.
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
- **맞춤형 씬 구성 (`DualArmSceneCfg`)**: 바닥, 두 대의 프랭카 로봇 팔, 초록색 큐브 물체, 그리고 빨간색 타겟 마커를 생성합니다.
- **포괄적인 관측 공간 (Observations)**: 에이전트가 큐브의 무작위 회전에 대응할 수 있도록 양팔의 손끝 3D 회전 각도(`pick_tcp_quat`, `place_tcp_quat`) 및 상대 거리가 관측 공간에 포함됩니다.
- **세분화된 보상 함수 (Dense Rewards)**: 복잡한 핸드오버 작업을 유도하고 꼼수(Exploits)를 완벽히 차단하기 위해 다음과 같이 구성되어 있습니다:
  - `action_penalty` (-0.01) & `action_rate_penalty` (-0.05): 거칠거나 떨리는 관절 움직임을 억제하여 부드러운 모션 유도.
  - `tcp_floor_penalty` (-5.0): 바닥 충돌 방지 페널티.
  - `pick_grasp_pose` (10.0): 자세 정렬 보상. 큐브의 얇은 면을 향해 손목을 완벽히 정렬하도록 유도하며, 그리퍼의 좌우 대칭 특성을 반영하여 180도 뒤집힌 그립도 만점을 주도록 수정됨.
  - `pick_reach` (20.0): **동적 목표점(Dynamic Target)** 방식을 도입하여, 큐브 정수리 위 12cm 상공으로 비행한 뒤 XY가 정렬되면 수직 하강하는 **'엘리베이터 궤적'**을 완벽하게 강제함. 최근 거리 마진을 4cm에서 1.5cm로 대폭 삭감하여 큐브 모서리만 꼬집는 얕은 그립(Edge Pinching) 꼼수를 원천 차단함.
  - `gripper_close` (20.0): 손을 닫는 보상. 지나치게 엄격했던 높이 제한을 해제하여 큐브의 윗부분을 쥐어도 보상을 받도록 완화.
  - `premature_gripper_close` (-0.5): 허공에서 미리 주먹을 쥐는 현상을 방지하되, 얼음(Freeze) 현상을 막기 위해 페널티 비중을 -0.5로 대폭 완화.
  - `pick_lift` (50.0): 물체를 바닥에서 들어 올렸을 때 부여되는 잭팟 보상.
  - `handover_approach` (20.0): 물체를 중앙 핸드오버 지점으로 가져올 때 주어지는 보상. 물체를 쳐서 허공에 띄운 뒤 빈 주먹을 쥐어 점수를 얻는 **"야구 배팅(Batting)" 꼼수 차단 로직** 추가.
  - `place_reach` (10.0): 왼쪽 팔이 핸드오버 지점으로 다가갈 때 주어지는 보상 (오른쪽 팔이 가져왔을 때만 활성화).
  - `place_grasp_pose` (10.0): 왼쪽 팔이 큐브를 측면에서 수평으로 접근하여 잡도록 자세를 유도하는 보상. (오른쪽 팔과의 충돌 방지)
  - `place_gripper_close` (20.0): 왼쪽 팔이 핸드오버 구역에서 물체를 건네받기 위해 주먹을 쥐었을 때 부여되는 보상.
  - `pick_release` (20.0): 왼쪽 팔이 물체를 꽉 잡은 것을 확인한 뒤, 오른쪽 팔이 그립을 열어 양보했을 때 주어지는 보상. 로봇들이 큐브를 양쪽에서 쥐고 놔주지 않는 **'줄다리기' 및 '동상' 현상 완벽 해결**.
  - `place_to_target` (100.0): 왼쪽 팔이 큐브를 들고 타겟을 향해 다가갈 때 주어지는 초밀집(Dense) 거리 비례 보상. 핸드오버 구역을 벗어날 때 발생하는 **'보상 계곡(Reward Valley)' 방어 로직**.
  - `place_object` (200.0): 최종 타겟에 안착했을 때 터지는 잭팟 보상. 불도저 꼼수를 막으면서도 **진정한 내려놓기(True Place)**를 유도하기 위해, 타겟 안착 후 양팔 모두 그립을 열고 멀리 물러났을 때만 점수를 주도록 재설계됨.
- **물리 엔진 최적화 & 조기 종료(Early Termination)**: 큐브에 고무 수준의 높은 마찰력(2.0)을 부여. 큐브가 책상 아래(`Z < -0.1`)로 떨어지면 에피소드를 즉시 리셋하여 허공에 헛손질하며 낭비되는 시간을 없애고 샘플 효율을 극대화함.
- **커리큘럼 학습 (Curriculum Learning)**: `curriculum_events.py`를 통해 큐브 스폰 난이도를 점진적으로 올림. (고정된 위치 -> 넓은 범위 & 360도 회전) 일반화 붕괴(Curriculum Shock) 방지.
- **대용량 신경망 (High-Capacity Neural Network)**: 듀얼 암의 복잡한 역운동학(IK) 매핑과 다단계 콤보 동작을 기억할 수 있도록 SKRL PPO 모델 용량을 `[1024, 512, 256]`으로 대폭 확장.
- **대규모 병렬 처리**: 수천 개의 환경을 동시에 실행하도록 구성되어 있습니다 (예: `num_envs=4096`).

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