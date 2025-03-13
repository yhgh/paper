import os
import json
import cv2
import numpy as np
import argparse
from scipy.spatial.transform import Rotation as Rot
from scipy.optimize import least_squares
from ultralytics import YOLO
import matplotlib
import matplotlib.pyplot as plt

# 使用中文宋体
matplotlib.rcParams['font.sans-serif'] = ['SimSun']  # 设置字体为宋体
matplotlib.rcParams['axes.unicode_minus'] = False    # 解决坐标轴负号显示问题

optimization_history = []

# ------------------- Camera -------------------
class Camera:
    fx = 0.0176  # [m]
    fy = 0.0176
    nu = 1920
    nv = 1200
    ppx = 5.86e-6
    ppy = ppx
    fpx = fx / ppx
    fpy = fy / ppy
    K = np.array([[fpx, 0, nu/2],
                  [0,   fpy, nv/2],
                  [0,   0,   1]])
    distCoeffs = np.array([-0.22383016606510672, 0.51409797089106379,
                           -0.00066499611998340662, -0.00021404771667484594,
                           -0.13124227429077406])
    cameraMatrix = np.array([
        [2988.5795163815555, 0, 960],
        [0, 2988.3401159176124, 600],
        [0, 0, 1]
    ])

#TRO优化目标函数 
def calculate_reprojection_error(kpts3d, kpts2d, rvec, tvec, cameraMatrix, distCoeffs):
    """计算重投影误差：3D点投影到2D平面后与实际2D点之间的平均欧氏距离"""
    # 重新投影3D点到2D平面
    reproj_points, _ = cv2.projectPoints(kpts3d, rvec, tvec, cameraMatrix, distCoeffs)
    reproj_points = reproj_points.reshape(-1, 2)  # 确保形状是 (N, 2)

    if kpts2d.ndim > 2:
        kpts2d = kpts2d.reshape(-1, 2)

    distances = np.linalg.norm(kpts2d - reproj_points, axis=1)

    error = np.mean(distances)
    return error

def objective_wrapper(params, kpts3d, kpts2d, cameraMatrix, distCoeffs, optimization_history):
    """目标函数包装器，用于记录优化过程中的误差历史"""
    rvec = params[:3].reshape((3, 1))
    tvec = params[3:].reshape((3, 1))
    
    # 计算当前参数的重投影误差
    error = calculate_reprojection_error(kpts3d, kpts2d, rvec, tvec, cameraMatrix, distCoeffs)
    
    # 记录当前迭代的误差
    optimization_history.append(error)
    
    # 计算目标函数值
    projected, _ = cv2.projectPoints(kpts3d, rvec, tvec, cameraMatrix, distCoeffs)
    projected = projected.reshape(-1, 2)
    
    if kpts2d.ndim > 2:
        kpts2d = kpts2d.reshape(-1, 2)
        
    obj_value = np.sum((projected - kpts2d)**2) / len(kpts3d)
    return obj_value

def run_tro_optimization(kpts3d, kpts2d, initial_rvec, initial_tvec, cameraMatrix, distCoeffs, 
                        max_nfev=100, method='trf', verbose=False):

    global optimization_history
    local_history = []
    
    # 初始参数(扁平化并拼接rvec和tvec)
    initial_params = np.concatenate([initial_rvec.flatten(), initial_tvec.flatten()])
    
    # 计算初始重投影误差
    initial_error = calculate_reprojection_error(kpts3d, kpts2d, initial_rvec, initial_tvec, cameraMatrix, distCoeffs)
    local_history.append(initial_error)
    
    # 使用least_squares和TRF方法运行优化
    result = least_squares(
        lambda params: objective_wrapper(params, kpts3d, kpts2d, cameraMatrix, distCoeffs, local_history),
        initial_params,
        method=method,  # 'trf'是信赖域反射算法
        max_nfev=max_nfev,
        verbose=2 if verbose else 0
    )
    
    # 提取优化后的旋转和平移向量
    rvec = result.x[:3].reshape((3, 1))
    tvec = result.x[3:].reshape((3, 1))
    
    return rvec, tvec, local_history

def quaternion_angle_error(pred_quat, true_quat):
    """
    计算两四元数的姿态角度差(度)。输入和内部计算均是 wxyz 顺序。
    """
    pred_quat = np.array(pred_quat, dtype=float)
    true_quat = np.array(true_quat, dtype=float)
    # 若点乘 < 0，则翻转 pred_quat 避免符号差异
    if np.dot(pred_quat, true_quat) < 0:
        pred_quat = -pred_quat
    pred_rot = Rot.from_quat([pred_quat[1], pred_quat[2], pred_quat[3], pred_quat[0]])
    true_rot = Rot.from_quat([true_quat[1], true_quat[2], true_quat[3], true_quat[0]])
    rel = pred_rot * true_rot.inv()
    ang = 2*np.arccos(np.clip(rel.as_quat()[-1], -1,1))
    deg = np.degrees(ang)
    if deg > 180:
        deg = 360 - deg
    return deg

def relative_position_error(tvec, true_pos):
    """
    位置相对误差: norm(tvec - true_pos) / norm(true_pos)
    """
    denom = np.linalg.norm(true_pos)
    if denom < 1e-12:
        return None
    return np.linalg.norm(tvec - true_pos) / denom

def normalize_quat(q):
    """ q=[w,x,y,z] """
    return q / np.linalg.norm(q)

def xyzQuat_to_matrix(rx, ry, rz, q_w, q_x, q_y, q_z):
    """将平移+四元数转换为 3x4 (R|t) 矩阵，供投影用"""
    R = Rot.from_quat([q_x, q_y, q_z, q_w]).as_matrix()  # 注意顺序: [x,y,z,w]
    t = np.array([rx, ry, rz], dtype=float).reshape(3, 1)
    Rt = np.hstack([R, t])
    return Rt

def project_points(X3d, Rt_3x4, camMat):
    """
    X3d: (N,3) in object coords
    => 先用 R|t => camera coords => pinhole => (u,v)
    返回 (N,2)
    """
    N = X3d.shape[0]
    ones = np.ones((N, 1), dtype=float)
    Xh = np.hstack([X3d, ones])        # (N,4)
    Xc = (Rt_3x4 @ Xh.T).T            # (N,3)
    # pinhole
    uvw = (camMat @ Xc.T).T           # (N,3)
    eps = 1e-12
    uvw[:,0] /= (uvw[:,2]+eps)
    uvw[:,1] /= (uvw[:,2]+eps)
    uvw[:,2] = 1.0
    return uvw[:,:2]

def quaternion_multiply(q1, q2):
    """ 四元数相乘，输入为 [w, x, y, z] """
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    w = w1*w2 - x1*x2 - y1*y2 - z1*z2
    x = w1*x2 + x1*w2 + y1*z2 - z1*y2
    y = w1*y2 - x1*z2 + y1*w2 + z1*x2
    z = w1*z2 + x1*y2 - y1*x2 + z1*w2
    return [w, x, y, z]

# =============== solvePnP 评估函数（返回 tvec 与四元数） ===============
def solve_pnp_and_error(kpts3d_obj, kpts2d_img, camera, true_quat, true_pos, use_trf=False, max_nfev=100):
    """
    调用 OpenCV solvePnP, 并额外返回 pnp 解出的 tvec, pred_quat (wxyz) 用于后续 EKF update。
    可选择使用TRF优化来进一步优化位姿。
    """
    kpts3d_obj = np.array(kpts3d_obj, dtype=np.float32).reshape(-1, 3)
    kpts2d_img = np.array(kpts2d_img, dtype=np.float32).reshape(-1, 2)
    if len(kpts3d_obj) < 4 or len(kpts3d_obj) != len(kpts2d_img):
        return (False, None, None, None, None, None)

    retval, rvec, tvec, inliers = cv2.solvePnPRansac(
        kpts3d_obj, kpts2d_img,
        camera.cameraMatrix,
        camera.distCoeffs,
        iterationsCount=10000,
        flags=cv2.SOLVEPNP_SQPNP
    )
    if not retval:
        return (False, None, None, None, None, None)
    
    if use_trf:
        if inliers is not None:
            kpts3d_inliers = kpts3d_obj[inliers.flatten()]
            kpts2d_inliers = kpts2d_img[inliers.flatten()]
        else:
            kpts3d_inliers = kpts3d_obj
            kpts2d_inliers = kpts2d_img
            
        # 运行TRF优化
        try:
            rvec, tvec, _ = run_tro_optimization(
                kpts3d_inliers, 
                kpts2d_inliers, 
                rvec, 
                tvec, 
                camera.cameraMatrix, 
                camera.distCoeffs, 
                max_nfev=max_nfev
            )
        except Exception as e:
            print(f"TRF优化失败: {e}")
    r_obj = Rot.from_rotvec(rvec.flatten())
    quat_xyzw = r_obj.as_quat()  # [x,y,z,w]
    pred_quat_wxyz = [quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]]

    ang_err = quaternion_angle_error(pred_quat_wxyz, true_quat)
    rel_err = None
    if true_pos is not None:
        rel_err = relative_position_error(tvec.flatten(), true_pos)

    inliers = inliers.flatten() if inliers is not None else np.arange(len(kpts3d_obj))
    proj_pts, _ = cv2.projectPoints(
        kpts3d_obj[inliers], rvec, tvec,
        camera.cameraMatrix, camera.distCoeffs
    )
    proj_pts = proj_pts.reshape(-1, 2)
    diff = proj_pts - kpts2d_img[inliers]
    reproj_err = float(np.mean(np.sqrt(np.sum(diff**2, axis=1))))

    return (True, ang_err, rel_err, reproj_err, tvec.flatten(), pred_quat_wxyz)

#YOLO 结果加载生成 
def generate_yolo_results(poses_data, images_dir, model):
    """
    若仍需要对图像做检测/关键点输出，可使用本函数。
    """
    results_dict = {}
    for pose in poses_data:
        fn = pose["filename"]
        img_path = os.path.join(images_dir, fn)
        img = cv2.imread(img_path)
        if img is None:
            continue
        res = model(img)
        if len(res) == 0 or len(res[0].keypoints) == 0:
            continue
        kdata = res[0].keypoints.data.cpu().numpy()[0]  # (N,3)
        kxy = res[0].keypoints.xy.cpu().numpy()[0]      # (N,2)
        confs = kdata[:, 2]
        results_dict[fn] = (kxy.tolist(), confs.tolist())
    return results_dict

# 6DoF EKF 
class RigidBody6DoFEKF_CV:
    """
    状态 x = [rx, ry, rz, qw, qx, qy, qz, vx, vy, vz, wx, wy, wz] (13维)
    过程模型:
        r_{k+1} = r_k + v_k * dt
        q_{k+1} = q_k ⊗ exp(ω_k * dt)
        v_{k+1} = v_k
        w_{k+1} = w_k

    测量模型: z = [rx, ry, rz, qw, qx, qy, qz]^T，直接用外部测量/估计(如PnP)得到的位姿更新。
    """

    def __init__(self, dt, q_scale=1.0, r_scale=1.0):
        self.dim_x = 13
        self.x = np.zeros((13, 1), dtype=float)
        # 初始姿态：qw=1
        self.x[3, 0] = 1.0

        # 初始协方差
        self.P = np.eye(self.dim_x) * 100.0

        # 过程噪声
        base_Q = np.eye(self.dim_x) * 0.01
        self.Q = base_Q * q_scale

        # 测量噪声缩放
        self.R_scale = r_scale

        self.dt = dt
        self.initialized = False

    def init_state(self, tvec, q_wxyz, v_init=np.zeros(3), w_init=np.zeros(3)):
        """ tvec=[x,y,z], q_wxyz=[w,x,y,z], v_init=[vx, vy, vz], w_init=[wx, wy, wz] """
        self.x[0,0] = tvec[0]
        self.x[1,0] = tvec[1]
        self.x[2,0] = tvec[2]
        qnorm = normalize_quat(q_wxyz)
        self.x[3,0] = qnorm[0]  # w
        self.x[4,0] = qnorm[1]  # x
        self.x[5,0] = qnorm[2]  # y
        self.x[6,0] = qnorm[3]  # z
        self.x[7:10,0] = v_init    # vx, vy, vz
        self.x[10:13,0] = w_init   # wx, wy, wz
        self.initialized = True

        self.P = np.eye(self.dim_x) * 0.1

    def predict(self):
        """ 
        r_{k+1} = r_k + v_k * dt
        q_{k+1} = q_k ⊗ exp( ω_k * dt )
        v_{k+1} = v_k
        w_{k+1} = w_k
        """
        if not self.initialized:
            return

        rx, ry, rz, qw, qx, qy, qz, vx, vy, vz, wx, wy, wz = self.x.flatten()

        # 位置
        self.x[0, 0] += vx * self.dt
        self.x[1, 0] += vy * self.dt
        self.x[2, 0] += vz * self.dt

        # 姿态
        omega = np.array([wx, wy, wz])
        omega_norm = np.linalg.norm(omega)
        if omega_norm > 1e-8:
            delta_theta = omega * self.dt
            delta_quat_xyzw = Rot.from_rotvec(delta_theta).as_quat()  # [x, y, z, w]
            delta_quat_wxyz = [delta_quat_xyzw[3], delta_quat_xyzw[0],
                               delta_quat_xyzw[1], delta_quat_xyzw[2]]
            current_quat = [qw, qx, qy, qz]
            new_quat = quaternion_multiply(current_quat, delta_quat_wxyz)
            new_quat = normalize_quat(new_quat)
            self.x[3, 0], self.x[4, 0], self.x[5, 0], self.x[6, 0] = new_quat

        # 更新协方差
        Fm = self.F()
        self.P = Fm @ self.P @ Fm.T + self.Q

    def F(self):
        """ 状态转移矩阵 (13x13), 匀速运动 """
        F = np.eye(self.dim_x)
        F[0,7] = self.dt
        F[1,8] = self.dt
        F[2,9] = self.dt
        return F

    def update_pose(self, tvec_meas, q_meas):

        if not self.initialized:
            return

        # 构造测量向量
        tvec_meas = np.array(tvec_meas, dtype=float).flatten()
        q_meas = normalize_quat(np.array(q_meas, dtype=float).flatten())
        z_meas = np.hstack([tvec_meas, q_meas]).reshape(-1,1)  # (7,1)

        # 预测量 h(x)
        h_x = self.x[0:7, :]  # [rx, ry, rz, qw, qx, qy, qz]^T

        # 测量残差
        y = z_meas - h_x

        # 测量雅可比 H: (7x13)
        H = np.zeros((7, 13), dtype=float)
        for i in range(7):
            H[i, i] = 1.0

        # 构造测量噪声 R (这里可根据实际需求调节)
        pos_sigma = 0.01 * self.R_scale
        quat_sigma = 0.01 * self.R_scale
        R = np.diag([pos_sigma**2, pos_sigma**2, pos_sigma**2,
                     quat_sigma**2, quat_sigma**2, quat_sigma**2, quat_sigma**2])

        # EKF Update
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)

        dx = K @ y
        self.x += dx

        # 归一化四元数
        qw_new, qx_new, qy_new, qz_new = self.x[3:7, 0]
        nq = normalize_quat([qw_new, qx_new, qy_new, qz_new])
        self.x[3, 0], self.x[4, 0], self.x[5, 0], self.x[6, 0] = nq

        # 更新协方差
        I = np.eye(self.dim_x)
        self.P = (I - K @ H) @ self.P

    def get_pose(self):
        rx, ry, rz, qw, qx, qy, qz, vx, vy, vz, wx, wy, wz = self.x.flatten()
        return np.array([rx, ry, rz]), np.array([qw, qx, qy, qz])

# =============== 6DoF EKF Pipeline with Constant Velocity (仅用位姿测量) ===============
def run_6dof_ekf_pipeline_cv(camera, poses_data, kpts3d_obj, yolo_results,
                            q_scale, r_scale, use_trf=False, max_nfev=100,
                            min_kpt_required=6, dt=1/30):
  

    ekf = RigidBody6DoFEKF_CV(dt=dt, q_scale=q_scale, r_scale=r_scale)

    raw_ang_list = []
    raw_pos_list = []
    raw_rep_list = []

    ekf_ang_list = []
    ekf_pos_list = []
    ekf_rep_list = []

    raw_ang_each_frame = []
    raw_pos_each_frame = []
    ekf_ang_each_frame = []
    ekf_pos_each_frame = []
    
    predicted_poses = []

    first_init = True

    for pose in poses_data:
        fn = pose["filename"]
        q_true = np.array(pose["q_vbs2tango_true"], dtype=float)
        r_true = pose.get("r_Vo2To_vbs_true", None)
        if r_true is not None:
            r_true = np.array(r_true, dtype=float)

     
        current_pred = {"filename": fn}
  
        for key, value in pose.items():
            if key not in ["filename", "q_vbs2tango_true", "r_Vo2To_vbs_true"]:
                current_pred[key] = value

        if fn not in yolo_results:
            raw_ang_each_frame.append(np.nan)
            raw_pos_each_frame.append(np.nan)
            ekf_ang_each_frame.append(np.nan)
            ekf_pos_each_frame.append(np.nan)
            

            predicted_poses.append(current_pred)
            continue

        kxy, confs = yolo_results[fn]
        kxy = np.array(kxy, dtype=float)
        confs = np.array(confs, dtype=float)

        idx_good = np.where(confs > 0.0)[0]
        if len(idx_good) < min_kpt_required:
            raw_ang_each_frame.append(np.nan)
            raw_pos_each_frame.append(np.nan)
            ekf_ang_each_frame.append(np.nan)
            ekf_pos_each_frame.append(np.nan)
            
            predicted_poses.append(current_pred)
            continue

        k3d_sel = kpts3d_obj[idx_good]
        k2d_sel = kxy[idx_good]

        # 未滤波 
        (ok, ang_pnp, pos_pnp, rep_pnp,
         tvec_pnp, quat_pnp_wxyz) = solve_pnp_and_error(
            k3d_sel, k2d_sel, camera, q_true, r_true, use_trf, max_nfev
        )
        if ok:
            raw_ang_list.append(ang_pnp)
            raw_rep_list.append(rep_pnp)
            raw_ang_each_frame.append(ang_pnp)
            if pos_pnp is not None:
                raw_pos_list.append(pos_pnp)
                raw_pos_each_frame.append(pos_pnp)
            else:
                raw_pos_each_frame.append(np.nan)
        else:
            # solvePnP 失败，记 NaN
            raw_ang_each_frame.append(np.nan)
            raw_pos_each_frame.append(np.nan)


        ekf.predict()
        if first_init:

            if r_true is not None:
                ekf.init_state(r_true, q_true)
            first_init = False
        else:
            if ok:

                ekf.update_pose(tvec_pnp, quat_pnp_wxyz)

        # 计算 EKF 当前误差
        e_r, e_q = ekf.get_pose()
        ang_ekf = quaternion_angle_error(e_q, q_true)
        pos_ekf = None
        if r_true is not None:
            pos_ekf = relative_position_error(e_r, r_true)

        # 计算 EKF 的重投影误差（仅做对比，不影响 EKF 更新）
        proj_ekf = project_points(
            k3d_sel,
            xyzQuat_to_matrix(e_r[0], e_r[1], e_r[2], e_q[0], e_q[1], e_q[2], e_q[3]),
            camera.cameraMatrix
        )
        diff = proj_ekf - k2d_sel
        rep_ekf = float(np.mean(np.sqrt(np.sum(diff**2, axis=1))))

        ekf_ang_list.append(ang_ekf)
        ekf_rep_list.append(rep_ekf)
        ekf_ang_each_frame.append(ang_ekf)
        if pos_ekf is not None:
            ekf_pos_list.append(pos_ekf)
            ekf_pos_each_frame.append(pos_ekf)
        else:
            ekf_pos_each_frame.append(np.nan)
            
        # 保存EKF预测的位姿
        if "q_vbs2tango_true" in pose:
            current_pred["q_vbs2tango_pred"] = e_q.tolist()
        if "r_Vo2To_vbs_true" in pose and r_true is not None:
            current_pred["r_Vo2To_vbs_pred"] = e_r.tolist()
            
        predicted_poses.append(current_pred)

    # 统计均值
    def m_or_none(arr):
        return np.mean(arr) if len(arr) > 0 else None

    raw_ang = m_or_none(raw_ang_list)
    raw_pos = m_or_none(raw_pos_list)
    raw_rep = m_or_none(raw_rep_list)

    ekf_ang = m_or_none(ekf_ang_list)
    ekf_pos = m_or_none(ekf_pos_list)
    ekf_rep = m_or_none(ekf_rep_list)

    return (raw_ang, raw_pos, raw_rep,
            ekf_ang, ekf_pos, ekf_rep,
            raw_ang_each_frame, raw_pos_each_frame,
            ekf_ang_each_frame, ekf_pos_each_frame,
            predicted_poses)

def save_predictions_to_json(predicted_poses, output_path):

    try:
        with open(output_path, 'w') as f:
            json.dump(predicted_poses, f, indent=2)
        print(f"成功保存预测位姿到: {output_path}")
        return True
    except Exception as e:
        print(f"保存预测位姿时出错: {e}")
        return False

# =============== 主流程 + 自动调参 + 可视化 ===============
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-yolo", action="store_true")
    parser.add_argument("--use-trf", action="store_true", help="是否使用TRF优化位姿估计")
    parser.add_argument("--max-nfev", type=int, default=100, help="TRF优化的最大函数评估次数")
    parser.add_argument("--save-predictions", action="store_true", default=True, 
                       help="是否保存预测位姿到JSON文件")
    args = parser.parse_args()

    with open("/home/ali213/Desktop/utils/blender_proc_scripts/kalmancmp/output_motion_dir.txt", "r") as f:
        output_motion_dir = f.readline().strip()
    casename = output_motion_dir.split("/")[-1]
    poses_data_file = os.path.join(output_motion_dir, "poses_data.json")
    images_dir = os.path.join(output_motion_dir, "images")

    kpts3d_path = "/home/ali213/Desktop/speedplusv2/synthetic/validation_data/3dloss_VitTpnet_starlink/kpts3d.npy"
    model_path = "/home/ali213/Desktop/speedplusv2/synthetic/validation_data/3dloss_VitTpnet_starlink/YOLOv8_n/weights/std.pt"
    yolo_cache_file = f"yolo_results_cache_{casename}.json"

    if not os.path.exists(poses_data_file):
        print("poses_data.json not found.")
        return
    if not os.path.exists(images_dir):
        print("images dir not found.")
        return
    if not os.path.exists(kpts3d_path):
        print("kpts3d not found.")
        return

    with open(poses_data_file, "r") as f:
        poses_data = json.load(f)
    if len(poses_data) == 0:
        print("No pose data.")
        return

    camera = Camera()
    kpts3d_obj = np.load(kpts3d_path).astype(np.float32)

    using_cache = True
    if using_cache:
        if not os.path.exists(yolo_cache_file):
            print(f"skip-yolo=True, but no {yolo_cache_file}")
            return
        with open(yolo_cache_file, "r") as f:
            yolo_results_raw = json.load(f)
        yolo_results = {}
        for fn, val in yolo_results_raw.items():
            kxy_list, conf_list = val
            yolo_results[fn] = (kxy_list, conf_list)
        print("[INFO] load yolo cache done.")
    else:
        if not os.path.exists(model_path):
            print("model not found.")
            return
        model = YOLO(model_path)
        print("[INFO] YOLO inferring...")
        yolo_results = generate_yolo_results(poses_data, images_dir, model)
        with open(yolo_cache_file, "w") as f:
            json.dump(yolo_results, f, indent=2)
        print("[INFO] YOLO done, saved cache.")

    # =========== 自动调参 (q_scale, r_scale) ===========
    q_candidates = [0.1]        # 可根据实际情况扩增
    r_candidates = [0.1]      # 位置/姿态测量噪声缩放

    best_score = 1e9
    best_params = None
    best_result = None
    best_predictions = None

    for q_s in q_candidates:
        for r_s in r_candidates:
            (raw_ang, raw_pos, raw_rep,
             ekf_ang, ekf_pos, ekf_rep,
             raw_ang_each_frame, raw_pos_each_frame,
             ekf_ang_each_frame, ekf_pos_each_frame,
             predicted_poses
            ) = run_6dof_ekf_pipeline_cv(
                camera, poses_data, kpts3d_obj, yolo_results,
                q_scale=q_s, r_scale=r_s, use_trf=args.use_trf, max_nfev=args.max_nfev,
                min_kpt_required=6, dt=1/30
            )

            if ekf_ang is None or ekf_pos is None or ekf_rep is None:
                continue

            score = np.radians(ekf_ang) + ekf_pos
            if score < best_score:
                best_score = score
                best_params = (q_s, r_s)
                best_result = (raw_ang, raw_pos, raw_rep,
                               ekf_ang, ekf_pos, ekf_rep,
                               raw_ang_each_frame, raw_pos_each_frame,
                               ekf_ang_each_frame, ekf_pos_each_frame)
                best_predictions = predicted_poses

            print(f"[*] q={q_s}, r={r_s}, raw_ang={raw_ang}, ekf_ang={ekf_ang}, score={score:.3f}")

    if best_params is None:
        print("[WARN] no valid params found.")
        return

    q_s, r_s = best_params
    (raw_ang, raw_pos, raw_rep,
     ekf_ang, ekf_pos, ekf_rep,
     raw_ang_each_frame, raw_pos_each_frame,
     ekf_ang_each_frame, ekf_pos_each_frame) = best_result

    print("\n========= 最优参数 (CV Model) =========")
    print(f"q_scale={q_s}, r_scale={r_s}")
    print("\n========= 对比结果 (CV Model) =========")
    if args.use_trf:
        print(f"[已启用TRF优化，max_nfev={args.max_nfev}]")
    print("[未滤波 solvePnP]:")
    print(f" 角度误差={raw_ang}, 相对位置误差={raw_pos}, 重投影误差={raw_rep}")
    raw_score = np.radians(raw_ang) + raw_pos
    print(f" 综合Score={raw_score:.3f}")
    print("[6DoF EKF CV Model] (仅位姿观测):")
    print(f" 角度误差={ekf_ang}, 相对位置误差={ekf_pos}, 重投影误差={ekf_rep}")
    print(f"综合Score={best_score:.3f}")

    # 保存预测结果到JSON文件
    if args.save_predictions and best_predictions is not None:
        pred_json_path = os.path.join(output_motion_dir, f"predict_{casename}EKF.json")
        save_predictions_to_json(best_predictions, pred_json_path)

    save_path = f"blender_proc_scripts/kalmancmp/cmpimg/{casename}/{casename}kalman_EKF.png"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    # 保存指标结果到txt文件
    txt_save_path = f"blender_proc_scripts/kalmancmp/cmpimg/{casename}/{casename}kalman_EKF.txt"
    with open(txt_save_path, 'w') as f:
        f.write(f"Raw Score={raw_score:.4f}\n")
        f.write(f"Filtered Score={best_score:.4f}\n")
        f.write(f"Raw angle error mean: {np.mean(raw_ang_each_frame):.4f}\n")
        f.write(f"Raw radians angle error mean: {np.mean(np.radians(raw_ang_each_frame)):.4f}\n")
        f.write(f"Filtered angle error mean: {np.mean(ekf_ang_each_frame):.4f}\n") 
        f.write(f"Filtered radians angle error mean: {np.mean(np.radians(ekf_ang_each_frame)):.4f}\n") 
        f.write(f"Raw position error mean: {np.mean(raw_pos_each_frame):.4f}\n")
        f.write(f"Filtered  position error mean: {np.mean(ekf_pos_each_frame):.4f}\n")
    # -------- 可视化：误差随帧变化 ---------
    frame_nums = np.arange(len(raw_ang_each_frame))

    plt.figure(figsize=(10,5))

    label_size = 29       
    tick_size = 25        
    title_size = 31       
    legend_size = 27     

    plt.figure(figsize=(10,5))

    # 子图1：角度误差
    plt.subplot(1,2,1)
    plt.plot(frame_nums, raw_ang_each_frame, linestyle='-', marker='o', color='red', linewidth=1.5, label='未滤波 角度误差')
    plt.plot(frame_nums, ekf_ang_each_frame, linestyle='--', marker='x', color='blue', linewidth=1.5, label='滤波后 角度误差')
    plt.xlabel('帧编号', fontsize=label_size)
    plt.ylabel('角度误差 (度)', fontsize=label_size)
    plt.title('角度误差随帧变化', fontsize=title_size)
    plt.legend(fontsize=legend_size)
    plt.grid(True)
    plt.tick_params(axis='both', which='major', labelsize=tick_size)  # 设置刻度值字体大小

    # 子图2：相对位置误差
    plt.subplot(1,2,2)
    plt.plot(frame_nums, raw_pos_each_frame, linestyle='-', marker='o', color='red', linewidth=1.5, label='未滤波 相对位置误差')
    plt.plot(frame_nums, ekf_pos_each_frame, linestyle='--', marker='x', color='blue', linewidth=1.5, label='滤波后 相对位置误差')
    plt.xlabel('帧编号', fontsize=label_size)
    plt.ylabel('相对位置误差', fontsize=label_size)
    plt.title('相对位置误差随帧变化', fontsize=title_size)
    plt.legend(fontsize=legend_size)
    plt.grid(True)
    plt.tick_params(axis='both', which='major', labelsize=tick_size)  # 设置刻度值字体大小

    plt.tight_layout()
   # 调整图形大小以匹配屏幕分辨率
    fig = plt.gcf()
    fig.set_size_inches(25.6, 16)  # 基于2560x1600的分辨率
    # 保存图像

    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    # plt.show()
    

if __name__ == "__main__":
    main()