import torch
from isaaclab.envs import ManagerBasedRLEnv

def pick_tcp_pos_w(env: ManagerBasedRLEnv, asset_name: str, pick_hand_regex: str) -> torch.Tensor:
    robot = env.scene[asset_name]
    wrist_idx = robot.find_bodies(pick_hand_regex)[0]
    wrist_pos = robot.data.body_pos_w[:, wrist_idx[0]]
    wrist_quat = robot.data.body_quat_w[:, wrist_idx[0]]
    w, x, y, z = wrist_quat[:, 0], wrist_quat[:, 1], wrist_quat[:, 2], wrist_quat[:, 3]
    z_dir_x = 2.0 * (x * z + w * y)
    z_dir_y = 2.0 * (y * z - w * x)
    z_dir_z = 1.0 - 2.0 * (x * x + y * y)
    z_dir = torch.stack([z_dir_x, z_dir_y, z_dir_z], dim=-1)
    tcp_pos = wrist_pos + 0.1034 * z_dir
    return tcp_pos

def place_tcp_pos_w(env: ManagerBasedRLEnv, asset_name: str, place_hand_regex: str) -> torch.Tensor:
    robot = env.scene[asset_name]
    wrist_idx = robot.find_bodies(place_hand_regex)[0]
    wrist_pos = robot.data.body_pos_w[:, wrist_idx[0]]
    wrist_quat = robot.data.body_quat_w[:, wrist_idx[0]]
    w, x, y, z = wrist_quat[:, 0], wrist_quat[:, 1], wrist_quat[:, 2], wrist_quat[:, 3]
    z_dir_x = 2.0 * (x * z + w * y)
    z_dir_y = 2.0 * (y * z - w * x)
    z_dir_z = 1.0 - 2.0 * (x * x + y * y)
    z_dir = torch.stack([z_dir_x, z_dir_y, z_dir_z], dim=-1)
    tcp_pos = wrist_pos + 0.1034 * z_dir
    return tcp_pos

def pick_tcp_quat_w(env: ManagerBasedRLEnv, asset_name: str, pick_hand_regex: str) -> torch.Tensor:
    robot = env.scene[asset_name]
    wrist_idx = robot.find_bodies(pick_hand_regex)[0]
    return robot.data.body_quat_w[:, wrist_idx[0]]

def place_tcp_quat_w(env: ManagerBasedRLEnv, asset_name: str, place_hand_regex: str) -> torch.Tensor:
    robot = env.scene[asset_name]
    wrist_idx = robot.find_bodies(place_hand_regex)[0]
    return robot.data.body_quat_w[:, wrist_idx[0]]

def object_to_pick_tcp_relative(env: ManagerBasedRLEnv, asset_name: str, pick_hand_regex: str, object_name: str) -> torch.Tensor:
    tcp_pos = pick_tcp_pos_w(env, asset_name, pick_hand_regex)
    obj = env.scene[object_name]
    obj_pos = obj.data.root_pos_w
    obj_quat = obj.data.root_quat_w
    
    ow, ox, oy, oz = obj_quat[:, 0], obj_quat[:, 1], obj_quat[:, 2], obj_quat[:, 3]
    cube_z_x = 2.0 * (ox * oz + ow * oy)
    cube_z_y = 2.0 * (oy * oz - ow * ox)
    cube_z_z = 1.0 - 2.0 * (ox * ox + oy * oy)
    cube_z_dir = torch.stack([cube_z_x, cube_z_y, cube_z_z], dim=-1)
    
    sign_y = torch.sign(cube_z_dir[:, 1])
    grab_pos = obj_pos - (sign_y.unsqueeze(-1) * 0.08 * cube_z_dir)
    return grab_pos - tcp_pos

def object_to_place_tcp_relative(env: ManagerBasedRLEnv, asset_name: str, place_hand_regex: str, object_name: str) -> torch.Tensor:
    tcp_pos = place_tcp_pos_w(env, asset_name, place_hand_regex)
    obj = env.scene[object_name]
    obj_pos = obj.data.root_pos_w
    obj_quat = obj.data.root_quat_w
    
    ow, ox, oy, oz = obj_quat[:, 0], obj_quat[:, 1], obj_quat[:, 2], obj_quat[:, 3]
    cube_z_x = 2.0 * (ox * oz + ow * oy)
    cube_z_y = 2.0 * (oy * oz - ow * ox)
    cube_z_z = 1.0 - 2.0 * (ox * ox + oy * oy)
    cube_z_dir = torch.stack([cube_z_x, cube_z_y, cube_z_z], dim=-1)
    
    sign_y = torch.sign(cube_z_dir[:, 1])
    grab_pos = obj_pos + (sign_y.unsqueeze(-1) * 0.08 * cube_z_dir)
    return grab_pos - tcp_pos

def object_to_target_relative(env: ManagerBasedRLEnv, object_name: str, target_name: str) -> torch.Tensor:
    obj = env.scene[object_name]
    target = env.scene[target_name]
    return target.data.root_pos_w - obj.data.root_pos_w
