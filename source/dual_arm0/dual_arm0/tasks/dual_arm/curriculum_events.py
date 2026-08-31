import torch
import math
from isaaclab.managers import SceneEntityCfg
from isaaclab.envs import ManagerBasedRLEnv
import sys


def reset_object_with_curriculum(env: ManagerBasedRLEnv, env_ids: torch.Tensor) -> dict:
    """커리큘럼(Curriculum)이 적용된 큐브 초기화 함수입니다.
    초반에는 고정된 위치에서 시작하여 학습 난이도를 낮추고,
    점진적으로 흩뿌리는 범위를 넓혀 최종적으로 넓은 범위와 무작위 회전을 소화하게 합니다.
    """

    # env.common_step_counter는 물리 스텝 수입니다.
    # 이를 기반으로 커리큘럼의 난이도(Progress)를 0.0에서 1.0까지 점진적으로 증가시킵니다.
    step = int(env.common_step_counter)

    # 2,000 스텝(PPO 기준 약 200 iter)부터 10,000 스텝에 걸쳐 난이도를 점진적으로 올림
    # 뇌 용량이 커졌으므로 튜토리얼 기간을 조금 더 넉넉하게 줍니다.
    progress = min(max((step - 2000) / 8000.0, 0.0), 1.0)

    # 평가 모드(play.py)이거나 체크포인트(--checkpoint)로 이어서 학습할 때는
    # 이미 튜토리얼을 뗐다고 간주하고 즉시 최고 난이도(1.0)를 적용합니다.
    # Hydra가 sys.argv를 수정하므로 sys.orig_argv를 사용합니다.
    if "play.py" in sys.argv[0] or any("--checkpoint" in arg for arg in getattr(sys, "orig_argv", sys.argv)):
        progress = 1.0

    print(
        f"[DEBUG 커리큘럼] 스텝: {step}, 진행도(progress): {progress:.2f}, 범위적용됨: {progress == 1.0}"
    )

    num_resets = len(env_ids)

    # 기본(쉬운) 위치 설정 (고정)
    base_x = 0.4
    base_y = -0.3
    base_roll = 0.0

    # 랜덤 노이즈 생성 (난이도가 0일 때는 노이즈도 0)
    if progress > 0.0:
        # X: -0.05m ~ +0.05m 범위
        x_noise = (torch.rand(num_resets, device=env.device) * 0.1 - 0.05) * progress
        # Y: -0.1m ~ +0.1m 범위
        y_noise = (torch.rand(num_resets, device=env.device) * 0.2 - 0.1) * progress
        # Roll: 0 ~ 360도 범위 (마지막에 폭발적으로 증가하도록 세제곱 적용)
        roll_noise = (torch.rand(num_resets, device=env.device) * 2 * math.pi) * (
            progress**3
        )
    else:
        x_noise = torch.zeros(num_resets, device=env.device)
        y_noise = torch.zeros(num_resets, device=env.device)
        roll_noise = torch.zeros(num_resets, device=env.device)

    final_x = base_x + x_noise
    final_y = base_y + y_noise
    final_roll = base_roll + roll_noise

    # Object 위치 덮어쓰기 로직
    obj = env.scene["object"]

    # 1. 시뮬레이터 상의 절대 좌표인 env_origins를 더해주어야 병렬 환경이 겹치지 않습니다.
    env_origins = env.scene.env_origins[env_ids]

    # 2. Position 설정 (env_origins 더하기)
    pos = obj.data.default_root_state[env_ids, :3].clone()
    pos[:, 0] = env_origins[:, 0] + final_x
    pos[:, 1] = env_origins[:, 1] + final_y
    # [수정] Z는 0.02(딱 바닥)일 경우 물리엔진 오차로 인해 바닥과 겹쳐서 공중으로 튀어오르는 현상이 생길 수 있으므로,
    # 0.025(2.5cm)로 약간 높여서 살짝 떨어지며 안정화되도록 합니다.
    pos[:, 2] = 0.025

    # 3. Rotation 설정 (Roll 적용)
    # 기본은 Y축으로 90도 누워있는 상태: (w=0.7071, x=0, y=0.7071, z=0)
    # 여기에 Roll 회전을 추가하기 위해 쿼터니언 곱셈 적용
    # Roll 축(로컬 X축) 회전 쿼터니언 계산: [cos(r/2), sin(r/2), 0, 0]
    roll_half = final_roll / 2.0
    rw = torch.cos(roll_half)
    rx = torch.sin(roll_half)

    # 기본 쿼터니언 q1 = [0.7071, 0, 0.7071, 0]
    q1_w = torch.full_like(rw, 0.7071)
    q1_x = torch.zeros_like(rw)
    q1_y = torch.full_like(rw, 0.7071)
    q1_z = torch.zeros_like(rw)

    # q1 * q2 쿼터니언 곱셈 수행
    qw = q1_w * rw - q1_x * rx
    qx = q1_w * rx + q1_x * rw
    qy = q1_y * rw + q1_z * rx
    qz = q1_z * rw - q1_y * rx

    rot = torch.stack([qw, qx, qy, qz], dim=-1)

    # 4. 시뮬레이션 상태 덮어쓰기 (write_root_pose_to_sim)
    obj.write_root_pose_to_sim(torch.cat([pos, rot], dim=-1), env_ids=env_ids)

    # 5. 루트 속도(Velocity) 0으로 초기화
    vel = torch.zeros((num_resets, 6), device=env.device)
    obj.write_root_velocity_to_sim(vel, env_ids=env_ids)

    return {}
