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
    
    obj_pos = obj.data.root_pos_w
    
    # 1. 물체 바로 위 12cm(0.12m, 큐브 높이의 3배) 지점을 '안전 대기(Hover) 위치'로 설정
    hover_pos = obj_pos.clone()
    hover_pos[:, 2] += 0.12
    dist_to_hover = torch.norm(tcp_pos - hover_pos, dim=-1)
    
    # (자세 정렬은 pick_grasp_pose_reward 에서 독립적으로 5.0점 만점으로 채점됨)
    
    # 3. 물체로의 하강 보상
    dist_to_obj = torch.norm(tcp_pos - obj_pos, dim=-1)
    
    # [수정] 큐브는 단단한 물체입니다. TCP가 큐브의 완벽한 정중앙(0cm)까지 파고들려고 하면 
    # 충돌(Collision)이 발생해 덜덜 떨게 됩니다. 
    # 반경 4cm(0.04m) 이내에 들어오면 거리를 0으로 간주해 만점을 주도록 마진(Margin)을 줍니다.
    dist_to_obj_clamped = torch.clamp(dist_to_obj - 0.04, min=0.0)
    
    reward_hover = 1.0 / (1.0 + 5.0 * dist_to_hover)
    approach_reward = 1.0 / (1.0 + 5.0 * dist_to_obj_clamped)
    
    # 호버 위치 도달(0.3)과 최종 하강(0.7)을 결합하여, 
    # 허공을 거쳐 큐브로 수직 하강하는 엘리베이터 궤적을 유도합니다.
    return 0.3 * reward_hover + 0.7 * approach_reward

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
    
    is_near_arm = dist < 0.10
    
    # [수정] 꼼수 방지: 주먹으로 큐브를 쳐서 위로 날려버리는(Batting/Flicking) 꼼수를 막기 위해,
    # 그리퍼가 닫혀있는지 확인합니다. 너무 타이트하면 살짝 미끄러졌을 때 보상을 못 받으므로 0.06m로 완화합니다.
    gripper_idx = robot.find_joints("panda_finger_joint.*")[0]
    gripper_pos = robot.data.joint_pos[:, gripper_idx]
    gripper_width = torch.sum(gripper_pos, dim=-1)
    is_closed = gripper_width < 0.06
    
    # [수정] 누워있는 상태(높이 0.02m)에서 시작하므로, 아주 미세하게라도(0.022m) 위로 들리면 점수를 주기 시작합니다.
    lift_amt = torch.clamp((obj_pos[:, 2] - 0.022) / 0.078, min=0.0, max=1.0)
    
    return lift_amt * is_near_arm.float() * is_closed.float()

def handover_zone_approach(env: ManagerBasedRLEnv, asset_name: str, pick_hand_regex: str, object_name: str, handover_pos: list) -> torch.Tensor:
    """물체가 들어 올려진 상태에서 중앙 핸드오버(인계) 구역으로 다가갈수록 보상을 줍니다."""
    robot = env.scene[asset_name]
    obj = env.scene[object_name]
    obj_pos = obj.data.root_pos_w
    
    target_pos = torch.tensor(handover_pos, device=env.device).unsqueeze(0)
    dist = torch.norm(obj_pos - target_pos, dim=-1)
    
    # 꼼수 방지: 물체가 허공에 떠 있더라도, 로봇이 꽉 쥐고(gripper_width < 0.05) 있을 때만 인정
    gripper_idx = robot.find_joints("panda_finger_joint.*")[0]
    gripper_pos = robot.data.joint_pos[:, gripper_idx]
    gripper_width = torch.sum(gripper_pos, dim=-1)
    is_closed = gripper_width < 0.05
    
    is_lifted = obj_pos[:, 2] > 0.1
    return torch.exp(-5.0 * dist) * is_lifted.float() * is_closed.float()

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

def gripper_close_reward(env: ManagerBasedRLEnv, asset_name: str, pick_hand_regex: str, object_name: str, gripper_joint_regex: str) -> torch.Tensor:
    """TCP가 물체 근처에 있을 때, 그리퍼(손가락)를 닫으면 강한 보상을 줍니다."""
    robot = env.scene[asset_name]
    obj = env.scene[object_name]
    
    # TCP 계산
    wrist_idx = robot.find_bodies(pick_hand_regex)[0]
    wrist_pos = robot.data.body_pos_w[:, wrist_idx[0]]
    wrist_quat = robot.data.body_quat_w[:, wrist_idx[0]]
    
    w, x, y, z = wrist_quat[:, 0], wrist_quat[:, 1], wrist_quat[:, 2], wrist_quat[:, 3]
    z_dir_x = 2.0 * (x * z + w * y)
    z_dir_y = 2.0 * (y * z - w * x)
    z_dir_z = 1.0 - 2.0 * (x * x + y * y)
    z_dir = torch.stack([z_dir_x, z_dir_y, z_dir_z], dim=-1)
    tcp_pos = wrist_pos + 0.1034 * z_dir
    
    obj_pos = obj.data.root_pos_w
    dist = torch.norm(tcp_pos - obj_pos, dim=-1)
    
    # 그리퍼 폭 계산
    gripper_idx, _ = robot.find_joints(gripper_joint_regex)
    gripper_pos = robot.data.joint_pos[:, gripper_idx]
    gripper_width = torch.sum(gripper_pos, dim=-1)
    
    # 자세 정렬도 계산
    # 1. 로컬 Y축 (손가락 방향) 계산
    robot_y_x = 2.0 * (x * y - w * z)
    robot_y_y = 1.0 - 2.0 * (x * x + z * z)
    robot_y_z = 2.0 * (y * z + w * x)
    
    # 2. 큐브의 로컬 Y축 (짧은 축) 계산
    obj_quat = obj.data.root_quat_w
    ow, ox, oy, oz = obj_quat[:, 0], obj_quat[:, 1], obj_quat[:, 2], obj_quat[:, 3]
    cube_y_x = 2.0 * (ox * oy - ow * oz)
    cube_y_y = 1.0 - 2.0 * (ox * ox + oz * oz)
    cube_y_z = 2.0 * (oy * oz + ow * ox)
    
    # 3. 내적(Dot Product)을 통해 평행도 계산
    dot_product = robot_y_x * cube_y_x + robot_y_y * cube_y_y + robot_y_z * cube_y_z
    pose_alignment = ((-z_dir_z + 1.0) / 2.0) * ((dot_product + 1.0) / 2.0)

    # 1. 포획(Engulfing) 조건: 거리를 조금 더 여유롭게 줍니다 (6cm 이내면 오르기 시작, 4cm 반경 내면 만점)
    # [추가] 큐브 윗면을 누르고 있으면서(거리는 가깝지만) 쥐는 꼼수를 막기 위해, 
    # TCP의 Z 높이가 큐브 중심점보다 확실하게 아래로(-0.01 등) 내려왔을 때만 포획으로 인정합니다.
    # (큐브 중심이 Z=0.02이므로, TCP가 Z=0.03 이하로 내려왔을 때만 인정)
    is_below_top = (tcp_pos[:, 2] < obj_pos[:, 2] + 0.01).float()
    is_engulfing = torch.clamp((0.06 - dist) / 0.02, 0.0, 1.0) * is_below_top
    
    # 2. 그리퍼 닫힘 조건: 4cm 이하로 닫히면 만점
    is_closed = torch.clamp((0.08 - gripper_width) / 0.04, 0.0, 1.0)
    
    # [수정] pose_alignment를 곱셈에서 분리하거나 제거합니다. 
    # 자세가 완벽하지 않더라도 쥐는 행위 자체에 보상을 주어 물체를 잡는 시도를 늘립니다.
    return is_engulfing * is_closed

def tcp_floor_collision_penalty(env: ManagerBasedRLEnv, asset_name: str, pick_hand_regex: str, object_name: str) -> torch.Tensor:
    """TCP(그리퍼 끝)가 바닥에 부딪히는 것을 방지하기 위해 물체 기본 높이의 1/2 밑으로 내려가면 페널티를 부과합니다."""
    robot = env.scene[asset_name]
    obj = env.scene[object_name]
    
    wrist_idx = robot.find_bodies(pick_hand_regex)[0]
    wrist_z = robot.data.body_pos_w[:, wrist_idx[0], 2]
    wrist_quat = robot.data.body_quat_w[:, wrist_idx[0]]
    
    # 쿼터니언을 통해 로컬 Z축의 월드 Z성분 계산
    x, y = wrist_quat[:, 1], wrist_quat[:, 2]
    z_dir_z = 1.0 - 2.0 * (x * x + y * y)
    
    # TCP의 Z 높이
    tcp_z = wrist_z + 0.1034 * z_dir_z
    
    # 물체의 기본 높이(Resting height) 가져오기
    # obj.data.default_root_state[:, 2]는 환경 초기화 시의 Z 좌표(즉, 큐브의 중심 높이)입니다.
    obj_base_z = obj.data.default_root_state[:, 2]
    min_height = 0.0  # [수정] 큐브 높이의 절반(1cm) 대신 진짜 바닥(0cm)으로 하한선 완화
    
    # min_height보다 아래로 내려간 깊이 (안 내려갔으면 0.0)
    violation = torch.clamp(min_height - tcp_z, min=0.0)
    
    return violation

def pick_grasp_pose_reward(env: ManagerBasedRLEnv, asset_name: str, pick_hand_regex: str, object_name: str) -> torch.Tensor:
    """TCP가 수직 아래를 바라보고, 손가락이 물체의 짧은 축(로컬 Y축)에 맞춰지도록 유도하는 보상"""
    robot = env.scene[asset_name]
    obj = env.scene[object_name]
    
    wrist_idx = robot.find_bodies(pick_hand_regex)[0]
    wrist_quat = robot.data.body_quat_w[:, wrist_idx[0]]
    obj_quat = obj.data.root_quat_w
    
    # 1. 로컬 Z축(수직 하강) -> 월드 -Z 방향 (-1.0 최고)
    w, x, y, z = wrist_quat[:, 0], wrist_quat[:, 1], wrist_quat[:, 2], wrist_quat[:, 3]
    z_dir_z = 1.0 - 2.0 * (x * x + y * y)
    vertical_alignment = (-z_dir_z + 1.0) / 2.0
    
    # 2. 로컬 Y축 (손가락 방향) 계산
    robot_y_x = 2.0 * (x * y - w * z)
    robot_y_y = 1.0 - 2.0 * (x * x + z * z)
    robot_y_z = 2.0 * (y * z + w * x)
    
    # 3. 큐브의 로컬 Y축 (짧은 축) 계산
    ow, ox, oy, oz = obj_quat[:, 0], obj_quat[:, 1], obj_quat[:, 2], obj_quat[:, 3]
    cube_y_x = 2.0 * (ox * oy - ow * oz)
    cube_y_y = 1.0 - 2.0 * (ox * ox + oz * oz)
    cube_y_z = 2.0 * (oy * oz + ow * ox)
    
    # 4. 두 Y축 간의 내적(Dot Product)을 통해 평행도 계산
    dot_product = robot_y_x * cube_y_x + robot_y_y * cube_y_y + robot_y_z * cube_y_z
    
    # 양방향 스피닝 방지를 위해 정방향 평행(+1.0)만 만점을 줌
    finger_alignment = (dot_product + 1.0) / 2.0
    
    # 두 정렬도를 곱하여 최종 자세 보상 반환
    return vertical_alignment * finger_alignment

def premature_gripper_close_penalty(env: ManagerBasedRLEnv, asset_name: str, pick_hand_regex: str, object_name: str, gripper_joint_regex: str) -> torch.Tensor:
    """큐브가 손가락 사이에 없는데도 미리 주먹을 쥐고 다가가서 큐브를 치고 다니는 행위를 방지하는 페널티입니다."""
    robot = env.scene[asset_name]
    obj = env.scene[object_name]
    
    # TCP 계산
    wrist_idx = robot.find_bodies(pick_hand_regex)[0]
    wrist_pos = robot.data.body_pos_w[:, wrist_idx[0]]
    wrist_quat = robot.data.body_quat_w[:, wrist_idx[0]]
    w, x, y, z = wrist_quat[:, 0], wrist_quat[:, 1], wrist_quat[:, 2], wrist_quat[:, 3]
    z_dir_x = 2.0 * (x * z + w * y)
    z_dir_y = 2.0 * (y * z - w * x)
    z_dir_z = 1.0 - 2.0 * (x * x + y * y)
    z_dir = torch.stack([z_dir_x, z_dir_y, z_dir_z], dim=-1)
    tcp_pos = wrist_pos + 0.1034 * z_dir
    
    obj_pos = obj.data.root_pos_w
    dist = torch.norm(tcp_pos - obj_pos, dim=-1)
    
    # 그리퍼 폭 계산
    gripper_idx, _ = robot.find_joints(gripper_joint_regex)
    gripper_pos = robot.data.joint_pos[:, gripper_idx]
    gripper_width = torch.sum(gripper_pos, dim=-1)
    
    # TCP가 큐브 윗면보다 높이 떠 있으면(Z축 기준) 큐브가 손가락 사이에 없다고 판단
    is_above_top = tcp_pos[:, 2] > obj_pos[:, 2] + 0.01
    
    # 큐브 중심에서 6cm보다 멀거나, 큐브 위를 누르고만 있으면 미포획 상태로 간주
    not_engulfing = torch.logical_or(dist > 0.06, is_above_top)
    
    # 그리퍼가 4cm(0.04m)보다 작게 열려 있으면 주먹을 쥐었다고 판단
    is_closed = gripper_width < 0.04
    
    # 큐브가 멀리 있거나 위에 얹혀 있는데 주먹을 쥐고 있으면 페널티 반환
    return (not_engulfing * is_closed).float()
