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
- **맞춤형 씬 구성 (`DualArmSceneCfg`)**: 바닥, 두 대의 프랭카 로봇 팔, 초록색 큐브 물체, 그리고 빨간색 타겟 마커를 생성합니다.
- **포괄적인 관측 공간 (Observations)**: 에이전트가 큐브의 무작위 회전에 대응할 수 있도록 양팔의 손끝 3D 회전 각도(`pick_tcp_quat`, `place_tcp_quat`) 및 상대 거리가 관측 공간에 포함됩니다.
- **세분화된 보상 함수 (Dense Rewards)**: 복잡한 핸드오버 작업을 유도하고 꼼수(Exploits)를 완벽히 차단하기 위해 다음과 같이 구성되어 있습니다:
  - `action_penalty` (-0.01) & `action_rate_penalty` (-0.05): 거칠거나 떨리는 관절 움직임을 억제하여 부드러운 모션 유도.
  - `tcp_floor_penalty` (-5.0): 바닥 충돌 방지 페널티.
  - `pick_grasp_pose` (10.0): 큐브가 바닥에 있을 때는 수직(Top-down) 자세를 강제하지만, 지면에서 5cm 이상 들리면 강제를 해제(무조건 만점 부여)하여 로봇이 팔을 뻗기 가장 편한 자세(Kinematic Singularity 회피)를 스스로 찾도록 자유도를 부여함.
  - `pick_reach` (20.0): **동적 목표점(Dynamic Target)** 방식을 도입하여, 큐브 정수리 위 12cm 상공으로 비행한 뒤 수직 하강하는 **'엘리베이터 궤적'** 강제.
  - `gripper_close` (20.0): 손을 닫는 보상.
  - `premature_gripper_close` (-0.5): 허공에서 미리 주먹을 쥐는 현상을 방지.
  - `pick_lift` (100.0): 물체를 바닥에서 들어 올렸을 때 부여되는 잭팟 보상. 큐브 무게를 1.0kg에서 0.2kg으로 경량화하여 안정적인 리프팅 성공.
  - `handover_approach` (200.0): 물체를 중앙 핸드오버 지점으로 가져올 때 보상. 인계점 높이를 40cm에서 20cm로 하향하여 오른팔의 물리적 리치(Reach) 한계 극복. 큐브가 미끄러지면 점수가 폭락하는 '떨림 공포증'을 치료하기 위해 잡기 판정 반경을 20cm에 걸쳐 서서히 깎이도록 극도로 관대하게 완화함.
  - `place_reach` (100.0): 왼쪽 팔이 핸드오버 지점으로 다가갈 때 주어지는 보상. 정규식 오타(`panda_hand` -> `panda_hand$`)로 인해 왼팔 보상을 오른팔이 독식하며 왼팔이 뇌사 상태에 빠졌던 치명적 버그 수정 완료.
  - `place_grasp_pose` (10.0): 왼쪽 팔이 큐브를 측면에서 수평으로 접근하여 잡도록 자세를 유도.
  - `place_gripper_close` (50.0): 왼쪽 팔이 핸드오버 구역에서 물체를 건네받기 위해 주먹을 쥐었을 때 보상.
  - `pick_release` (100.0): 왼쪽 팔이 물체를 꽉 잡은 것을 확인한 뒤, 오른쪽 팔이 그립을 열어 양보했을 때 주어지는 보상.
  - `place_to_target` (200.0): 왼쪽 팔이 큐브를 들고 타겟을 향해 다가갈 때 주어지는 초밀집(Dense) 거리 비례 보상.
  - `place_object` (1000.0): 최종 타겟에 안착했을 때 터지는 잭팟 보상.
- **물리 엔진 최적화 & 조기 종료(Early Termination)**: 큐브 무게(0.2kg) 최적화 및 고마찰력(2.0) 적용. 큐브가 책상 아래(`Z < -0.1`)로 떨어지면 에피소드를 즉시 리셋.
- **커리큘럼 학습 (Curriculum Learning)**: `curriculum_events.py`를 통해 큐브 스폰 난이도를 점진적으로 올림. (고정된 위치 -> 넓은 범위 & 360도 회전) 체크포인트(`--checkpoint`)로 이어서 학습할 경우, 튜토리얼 단계를 건너뛰고 즉시 최고 난이도(progress=1.0)의 무작위 스폰이 적용되도록 개선되었습니다.
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