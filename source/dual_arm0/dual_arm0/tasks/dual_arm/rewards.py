import torch
from isaaclab.managers import SceneEntityCfg
from isaaclab.envs import ManagerBasedRLEnv

def pick_reach_object(env: ManagerBasedRLEnv, asset_name: str, pick_hand_regex: str, object_name: str) -> torch.Tensor:
    """잡는 팔(Pick Arm)의 손끝(TCP)이 물체에 가까워질수록 보상을 줍니다.
    
    Args:
        env (ManagerBasedRLEnv): Isaac Lab RL 환경 인스턴스.
        asset_name (str): 로봇 에셋의 이름.
        pick_hand_regex (str): 손목(wrist)이 아닌 손가락을 찾기 위한 기본 문자열 (실제로는 내부에서 finger 링크를 찾습니다).
        object_name (str): 조작 대상 물체의 이름.
    """
    robot = env.scene[asset_name]
    obj = env.scene[object_name]
    
    # 정규표현식을 매칭하여 손목 링크 인덱스 획득
    wrist_idx = robot.find_bodies(pick_hand_regex)[0]
    
    # 손목의 위치와 회전(쿼터니언) 획득
    wrist_pos = robot.data.body_pos_w[:, wrist_idx[0]]
    wrist_quat = robot.data.body_quat_w[:, wrist_idx[0]]
    
    # Franka 로봇의 경우 손목(panda_hand)에서 실제 손가락 끝(TCP)까지 Z축으로 약 10.34cm 떨어져 있습니다.
    # 손목 좌표계 기준 Z축 방향으로 0.1034m 떨어진 곳의 월드 좌표를 계산합니다.
    # Z축 로컬 벡터 (0, 0, 1)을 월드 좌표계로 변환 (쿼터니언 회전 적용)
    # PyTorch batch 연산을 위해 다음과 같이 계산합니다.
    # q = [w, x, y, z] in Isaac Sim
    w, x, y, z = wrist_quat[:, 0], wrist_quat[:, 1], wrist_quat[:, 2], wrist_quat[:, 3]
    # Z축 방향 벡터 계산 (쿼터니언에 (0,0,1) 적용)
    z_dir_x = 2.0 * (x * z + w * y)
    z_dir_y = 2.0 * (y * z - w * x)
    z_dir_z = 1.0 - 2.0 * (x * x + y * y)
    z_dir = torch.stack([z_dir_x, z_dir_y, z_dir_z], dim=-1)
    
    # TCP (손끝 중앙) 위치 = 손목 위치 + 10.34cm * Z방향
    tcp_pos = wrist_pos + 0.1034 * z_dir
    
    obj_pos = obj.data.root_pos_w  # 물체의 월드 기준 루트 위치
    
    # TCP와 물체 사이의 L2 거리를 구합니다.
    dist = torch.norm(tcp_pos - obj_pos, dim=-1)
    
    # 거리가 0에 가까워질수록 1에 수렴하는 보상을 반환합니다.
    return torch.exp(-10.0 * dist)

def object_lifted_by_pick_arm(env: ManagerBasedRLEnv, asset_name: str, pick_hand_regex: str, object_name: str) -> torch.Tensor:
    """잡는 팔(Pick Arm) 부근에서 물체가 바닥으로부터 일정 높이 이상 들려 올려졌을 때 보상을 줍니다."""
    robot = env.scene[asset_name]
    obj = env.scene[object_name]
    
    wrist_idx = robot.find_bodies(pick_hand_regex)[0]
    wrist_pos = robot.data.body_pos_w[:, wrist_idx[0]]
    wrist_quat = robot.data.body_quat_w[:, wrist_idx[0]]
    
    # TCP 계산 (손끝 10.34cm 앞)
    w, x, y, z = wrist_quat[:, 0], wrist_quat[:, 1], wrist_quat[:, 2], wrist_quat[:, 3]
    z_dir_x = 2.0 * (x * z + w * y)
    z_dir_y = 2.0 * (y * z - w * x)
    z_dir_z = 1.0 - 2.0 * (x * x + y * y)
    z_dir = torch.stack([z_dir_x, z_dir_y, z_dir_z], dim=-1)
    tcp_pos = wrist_pos + 0.1034 * z_dir
    
    obj_pos = obj.data.root_pos_w
    dist = torch.norm(tcp_pos - obj_pos, dim=-1)
    
    # TCP(손끝)가 10cm 이내에 있을 때, 물체의 높이가 0.06m 이상으로 올라갈수록 연속적인 보상을 줍니다.
    # (원기둥이 쓰러질 때 무게중심이 최대 5.6cm까지 높아지는 '꼼수(Exploit)'를 방지하기 위해 기준을 6cm로 상향합니다!)
    is_near_arm = dist < 0.10
    
    # 높이가 0.06m에서 0.10m로 올라갈 때 0.0 ~ 1.0 사이의 값이 되도록 정규화
    lift_amt = torch.clamp((obj_pos[:, 2] - 0.06) / 0.04, min=0.0, max=1.0)
    
    # 0과 1 사이로 부드럽게 증가하는 보상 반환
    return lift_amt * is_near_arm.float()

def handover_zone_approach(env: ManagerBasedRLEnv, object_name: str, handover_pos: list) -> torch.Tensor:
    """물체가 들어 올려진 상태에서 중앙 핸드오버(인계) 구역으로 다가갈수록 보상을 줍니다."""
    obj = env.scene[object_name]
    obj_pos = obj.data.root_pos_w
    
    target_pos = torch.tensor(handover_pos, device=env.device).unsqueeze(0)
    dist = torch.norm(obj_pos - target_pos, dim=-1)
    
    is_lifted = obj_pos[:, 2] > 0.1
    return torch.exp(-5.0 * dist) * is_lifted.float()

def place_reach_object(env: ManagerBasedRLEnv, asset_name: str, place_hand_regex: str, object_name: str, handover_pos: list) -> torch.Tensor:
    """물체가 핸드오버 구역 내에 있을 때만, 내려놓는 팔(Place Arm) 그리퍼가 물체에 가까워질수록 보상을 줍니다."""
    robot = env.scene[asset_name]
    obj = env.scene[object_name]
    
    wrist_idx = robot.find_bodies(place_hand_regex)[0]
    wrist_pos = robot.data.body_pos_w[:, wrist_idx[0]]
    wrist_quat = robot.data.body_quat_w[:, wrist_idx[0]]
    
    # TCP 계산
    w, x, y, z = wrist_quat[:, 0], wrist_quat[:, 1], wrist_quat[:, 2], wrist_quat[:, 3]
    z_dir_x = 2.0 * (x * z + w * y)
    z_dir_y = 2.0 * (y * z - w * x)
    z_dir_z = 1.0 - 2.0 * (x * x + y * y)
    z_dir = torch.stack([z_dir_x, z_dir_y, z_dir_z], dim=-1)
    tcp_pos = wrist_pos + 0.1034 * z_dir
    
    obj_pos = obj.data.root_pos_w
    
    target_pos = torch.tensor(handover_pos, device=env.device).unsqueeze(0)
    dist_to_handover = torch.norm(obj_pos - target_pos, dim=-1)
    is_in_zone = dist_to_handover < 0.15
    
    dist_to_tcp = torch.norm(tcp_pos - obj_pos, dim=-1)
    return torch.exp(-10.0 * dist_to_tcp) * is_in_zone.float()

def object_to_target(env: ManagerBasedRLEnv, asset_name: str, place_hand_regex: str, object_name: str, target_name: str) -> torch.Tensor:
    """물체가 내려놓는 팔(Place Arm) 근처에 파지되어 있을 때만, 물체를 최종 목표 지점으로 이동시킬수록 보상을 줍니다."""
    obj = env.scene[object_name]
    target = env.scene[target_name]
    robot = env.scene[asset_name]
    
    wrist_idx = robot.find_bodies(place_hand_regex)[0]
    wrist_pos = robot.data.body_pos_w[:, wrist_idx[0]]
    wrist_quat = robot.data.body_quat_w[:, wrist_idx[0]]
    
    # TCP 계산
    w, x, y, z = wrist_quat[:, 0], wrist_quat[:, 1], wrist_quat[:, 2], wrist_quat[:, 3]
    z_dir_x = 2.0 * (x * z + w * y)
    z_dir_y = 2.0 * (y * z - w * x)
    z_dir_z = 1.0 - 2.0 * (x * x + y * y)
    z_dir = torch.stack([z_dir_x, z_dir_y, z_dir_z], dim=-1)
    tcp_pos = wrist_pos + 0.1034 * z_dir
    
    obj_pos = obj.data.root_pos_w
    dist_to_tcp = torch.norm(tcp_pos - obj_pos, dim=-1)
    
    dist_to_target_2d = torch.norm(obj_pos[:, :2] - target.data.root_pos_w[:, :2], dim=-1)
    
    is_near_arm = dist_to_tcp < 0.10
    return torch.exp(-5.0 * dist_to_target_2d) * is_near_arm.float()
