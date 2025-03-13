import os
import json
import cv2
import numpy as np
import argparse
import matplotlib
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
import time

# 使用中文宋体
matplotlib.rcParams['font.sans-serif'] = ['SimSun']  # 设置字体为宋体
matplotlib.rcParams['axes.unicode_minus'] = False    # 解决坐标轴负号显示问题


optimization_history = []

# 李群/李代数辅助函数
def hat(v):

    return np.array([
        [0,     -v[2],  v[1]],
        [v[2],   0,    -v[0]],
        [-v[1],  v[0],  0   ]
    ], dtype=float)

def exp_so3(phi):

    theta = np.linalg.norm(phi)
    if theta < 1e-12:
        return np.eye(3) + hat(phi)
    else:
        axis = phi / theta
        K = hat(axis)
        return (np.eye(3)
                + np.sin(theta)*K
                + (1 - np.cos(theta))*(K @ K))

def log_so3(R):
    tr = np.trace(R)
    cos_theta = 0.5*(tr - 1)
    if cos_theta > 1.0:
        cos_theta = 1.0
    elif cos_theta < -1.0:
        cos_theta = -1.0
    theta = np.arccos(cos_theta)
    if abs(theta) < 1e-12:
        return np.zeros(3)
    tmp = (R - R.T)/(2*np.sin(theta))
    return np.array([tmp[2,1], tmp[0,2], tmp[1,0]]) * theta

def rotation_matrix_to_quaternion(R):

    m00, m01, m02 = R[0,0], R[0,1], R[0,2]
    m10, m11, m12 = R[1,0], R[1,1], R[1,2]
    m20, m21, m22 = R[2,0], R[2,1], R[2,2]
    trace = m00 + m11 + m22

    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (m21 - m12) * s
        y = (m02 - m20) * s
        z = (m10 - m01) * s
    else:
        if m00 > m11 and m00 > m22:
            s = 2.0 * np.sqrt(1.0 + m00 - m11 - m22)
            w = (m21 - m12) / s
            x = 0.25 * s
            y = (m01 + m10) / s
            z = (m02 + m20) / s
        elif m11 > m22:
            s = 2.0 * np.sqrt(1.0 + m11 - m00 - m22)
            w = (m02 - m20) / s
            x = (m01 + m10) / s
            y = 0.25 * s
            z = (m12 + m21) / s
        else:
            s = 2.0 * np.sqrt(1.0 + m22 - m00 - m11)
            w = (m10 - m01) / s
            x = (m02 + m20) / s
            y = (m12 + m21) / s
            z = 0.25 * s

    q = np.array([w, x, y, z])
    q /= np.linalg.norm(q)
    return q

def quat_to_matrix(q):

    w, x, y, z = q
    # 归一化
    n = np.sqrt(w*w + x*x + y*y + z*z)
    if n < 1e-15:
        return np.eye(3)
    w, x, y, z = w/n, x/n, y/n, z/n

    R = np.array([
        [1 - 2*y*y - 2*z*z,     2*x*y - 2*w*z,         2*x*z + 2*w*y],
        [2*x*y + 2*w*z,         1 - 2*x*x - 2*z*z,     2*y*z - 2*w*x],
        [2*x*z - 2*w*y,         2*y*z + 2*w*x,         1 - 2*x*x - 2*y*y]
    ], dtype=float)
    return R

def rotation_angle_error(R_pred, R_true):

    R_rel = R_pred.T @ R_true
    tr = np.trace(R_rel)
    val = (tr - 1.0)/2.0
    # 数值稳定
    if val > 1.0:
        val = 1.0
    elif val < -1.0:
        val = -1.0
    angle = np.arccos(val)
    deg = np.degrees(angle)
    if deg > 180:
        deg = 360 - deg
    return deg

# ------------------- TRF优化相关函数 -------------------
def calculate_reprojection_error(kpts3d, kpts2d, rvec, tvec, cameraMatrix, distCoeffs):
    reproj_points, _ = cv2.projectPoints(kpts3d, rvec, tvec, cameraMatrix, distCoeffs)
    reproj_points = reproj_points.reshape(-1, 2)  # 确保形状是 (N, 2)

    if kpts2d.ndim > 2:
        kpts2d = kpts2d.reshape(-1, 2)

    distances = np.linalg.norm(kpts2d - reproj_points, axis=1)

    error = np.mean(distances)
    return error

def objective_wrapper(params, kpts3d, kpts2d, cameraMatrix, distCoeffs, optimization_history):
    rvec = params[:3].reshape((3, 1))
    tvec = params[3:].reshape((3, 1))
    
    error = calculate_reprojection_error(kpts3d, kpts2d, rvec, tvec, cameraMatrix, distCoeffs)
    
    optimization_history.append(error)
    
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


def relative_position_error(tvec, true_pos):
    """
    位置相对误差
    """
    denom = np.linalg.norm(true_pos)
    if denom < 1e-12:
        return None
    return np.linalg.norm(tvec - true_pos) / denom

def project_points(X3d, Rt_3x4, camMat):
    N = X3d.shape[0]
    ones = np.ones((N, 1), dtype=float)
    Xh = np.hstack([X3d, ones])        # (N,4)
    Xc = (Rt_3x4 @ Xh.T).T            # (N,3)
    # pinhole
    uvw = (camMat @ Xc.T).T           # (N,3)
    eps = 1e-12
    uvw[:,0] /= (uvw[:,2] + eps)
    uvw[:,1] /= (uvw[:,2] + eps)
    uvw[:,2] = 1.0
    return uvw[:, :2]


def xyzPhi_to_matrix(rx, ry, rz, phi_x, phi_y, phi_z):

    phi = np.array([phi_x, phi_y, phi_z], dtype=float)
    R = exp_so3(phi)
    t = np.array([rx, ry, rz], dtype=float).reshape(3, 1)
    Rt = np.hstack([R, t])
    return Rt

def solve_pnp_and_error(kpts3d_obj, kpts2d_img, camera, true_quat, true_pos, use_trf=False, max_nfev=100):
    
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

    R_pred, _ = cv2.Rodrigues(rvec)  # 3x3
    R_true = quat_to_matrix(true_quat)

    ang_err = rotation_angle_error(R_pred, R_true)

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

    phi = log_so3(R_pred)

    return (True, ang_err, rel_err, reproj_err, tvec.flatten(), phi)

def generate_yolo_results(poses_data, images_dir, model):
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

class RigidBody6DoFEKF_CV_Lie:


    def __init__(self, dt, q_scale=1.0, r_scale=1.0):
        self.dim_x = 12
        # x = [rx, ry, rz, phi_x, phi_y, phi_z, vx, vy, vz, wx, wy, wz]
        self.x = np.zeros((self.dim_x, 1), dtype=float)
        self.P = np.eye(self.dim_x) * 100.0  # 初始协方差
        self.dt = dt

        # 过程噪声
        base_Q = np.eye(self.dim_x) * 0.01
        self.Q = base_Q * q_scale

        # 量测噪声 R 的缩放
        self.R_scale = r_scale

        self.initialized = False

    def init_state(self, tvec, q_wxyz, v_init=np.zeros(3), w_init=np.zeros(3)):
      
        self.x[0,0] = tvec[0]
        self.x[1,0] = tvec[1]
        self.x[2,0] = tvec[2]

        R_init = quat_to_matrix(q_wxyz)
        phi_init = log_so3(R_init)
        self.x[3,0] = phi_init[0]
        self.x[4,0] = phi_init[1]
        self.x[5,0] = phi_init[2]

        self.x[6:9,0] = v_init
        self.x[9:12,0] = w_init

        self.initialized = True
        
    def init_state_with_pose(self, tvec, phi, v_init=np.zeros(3), w_init=np.zeros(3)):
        
        self.x[0,0] = tvec[0]
        self.x[1,0] = tvec[1]
        self.x[2,0] = tvec[2]
        self.x[3,0] = phi[0]
        self.x[4,0] = phi[1]
        self.x[5,0] = phi[2]
        self.x[6:9,0] = v_init
        self.x[9:12,0] = w_init
        self.initialized = True

    def F(self):
        F = np.eye(self.dim_x)
        dt = self.dt
        # dr/dv
        F[0, 6] = dt
        F[1, 7] = dt
        F[2, 8] = dt
        # dphi/dw
        F[3,  9] = dt
        F[4, 10] = dt
        F[5, 11] = dt
        return F

    def predict(self):
       
        if not self.initialized:
            return

        rx, ry, rz = self.x[0:3,0]
        phi_x, phi_y, phi_z = self.x[3:6,0]
        vx, vy, vz = self.x[6:9,0]
        wx, wy, wz = self.x[9:12,0]

        dt = self.dt
        # 更新位置
        self.x[0,0] = rx + vx*dt
        self.x[1,0] = ry + vy*dt
        self.x[2,0] = rz + vz*dt

        # 更新姿态 (小角度近似)
        self.x[3,0] = phi_x + wx*dt
        self.x[4,0] = phi_y + wy*dt
        self.x[5,0] = phi_z + wz*dt

        # 线速度和角速度不变

        # 更新协方差
        Fm = self.F()
        self.P = Fm @ self.P @ Fm.T + self.Q

    def update(self, kpts3d_obj, kpts2d, camMat):
        if not self.initialized:
            return

        N = kpts3d_obj.shape[0]
        z_meas = kpts2d.reshape(-1, 1)  # (2N,1)

        # 当前预测投影 z_pred
        rx, ry, rz = self.x[0:3,0]
        phi_x, phi_y, phi_z = self.x[3:6,0]
        Rt_3x4 = xyzPhi_to_matrix(rx, ry, rz, phi_x, phi_y, phi_z)
        z_pred_2d = project_points(kpts3d_obj, Rt_3x4, camMat)  # (N,2)
        z_pred = z_pred_2d.reshape(-1, 1)  # (2N,1)

        # 数值差分计算 H (2N x 12)
        eps = 1e-5
        H = np.zeros((2*N, self.dim_x), dtype=float)

        for j in range(self.dim_x):
            x0 = self.x[j, 0]
            # 正向扰动
            self.x[j, 0] = x0 + eps

            # 重新计算投影
            rx_p, ry_p, rz_p = self.x[0:3,0]
            phx_p, phy_p, phz_p = self.x[3:6,0]
            Rt_p = xyzPhi_to_matrix(rx_p, ry_p, rz_p, phx_p, phy_p, phz_p)
            z_plus_2d = project_points(kpts3d_obj, Rt_p, camMat).reshape(-1, 1)

            # 恢复
            self.x[j, 0] = x0

            # 数值差分
            dz = (z_plus_2d - z_pred)/eps
            H[:, j] = dz.flatten()

        # 量测噪声 R：假设像素级噪声 ~ 5*r_scale
        sigma = 5.0 * self.R_scale
        R = np.eye(2*N) * (sigma**2)

        # EKF Update
        y = z_meas - z_pred  # Innovation
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)
        dx = K @ y

        # 状态更新
        self.x += dx

        # 更新协方差
        I = np.eye(self.dim_x)
        self.P = (I - K @ H) @ self.P

    def get_pose(self):
     
        rx, ry, rz = self.x[0:3,0]
        phi = self.x[3:6,0]
        R = exp_so3(phi)
        q = rotation_matrix_to_quaternion(R)  # (w, x, y, z)
        return np.array([rx, ry, rz]), q

# =============== 6DoF EKF Pipeline with Constant Velocity (Lie Group) ===============
def run_6dof_ekf_pipeline_cv_lie(camera, poses_data, kpts3d_obj, yolo_results,
                                 q_scale, r_scale, use_trf=False, max_nfev=100,
                                 min_kpt_required=6, dt=1/30):
   
    import time
    ekf = RigidBody6DoFEKF_CV_Lie(dt=dt, q_scale=q_scale, r_scale=r_scale)

    raw_ang_list = []
    raw_pos_list = []
    raw_rep_list = []

    ekf_ang_list = []
    ekf_pos_list = []
    ekf_rep_list = []

    raw_ang_each_frame = []
    raw_pos_each_frame = []
    raw_rep_each_frame = []

    ekf_ang_each_frame = []
    ekf_pos_each_frame = []
    ekf_rep_each_frame = []
    
    timing_info = {
        "pnp_times": [], 
        "ekf_times": [],
        "total_times": []
    }
    
    output_poses = []

    first_init = True

    for pose in poses_data:
        fn = pose["filename"]
        time_stamp = pose.get("time", 0.0)
        q_true = np.array(pose["q_vbs2tango_true"], dtype=float)  # [w, x, y, z]
        r_true = pose.get("r_Vo2To_vbs_true", None)
        if r_true is not None:
            r_true = np.array(r_true, dtype=float)

        if fn not in yolo_results:
            raw_ang_each_frame.append(np.nan)
            raw_pos_each_frame.append(np.nan)
            raw_rep_each_frame.append(np.nan)
            ekf_ang_each_frame.append(np.nan)
            ekf_pos_each_frame.append(np.nan)
            ekf_rep_each_frame.append(np.nan)
            
            # 添加原始位姿数据
            output_poses.append({
                "time": time_stamp,
                "r_Vo2To_vbs_true": r_true.tolist() if r_true is not None else None,
                "q_vbs2tango_true": q_true.tolist(),
                "filename": fn
            })
            continue

        kxy, confs = yolo_results[fn]
        kxy = np.array(kxy, dtype=float)
        confs = np.array(confs, dtype=float)

        idx_good = np.where(confs > 0.0)[0]
        if len(idx_good) < min_kpt_required:
            raw_ang_each_frame.append(np.nan)
            raw_pos_each_frame.append(np.nan)
            raw_rep_each_frame.append(np.nan)
            ekf_ang_each_frame.append(np.nan)
            ekf_pos_each_frame.append(np.nan)
            ekf_rep_each_frame.append(np.nan)
            
            # 添加原始位姿数据
            output_poses.append({
                "time": time_stamp,
                "r_Vo2To_vbs_true": r_true.tolist() if r_true is not None else None,
                "q_vbs2tango_true": q_true.tolist(),
                "filename": fn
            })
            continue

        # 未滤波
        k3d_sel = kpts3d_obj[idx_good]
        k2d_sel = kxy[idx_good]
        
        start_pnp_time = time.time()
        ok, ang_pnp, pos_pnp, rep_pnp, tvec_pnp, phi_pnp = solve_pnp_and_error(
            k3d_sel, k2d_sel, camera, q_true, r_true, use_trf, max_nfev
        )
        pnp_time = time.time() - start_pnp_time
        timing_info["pnp_times"].append(pnp_time)
        
        if ok:
            R_pnp = exp_so3(phi_pnp)
            q_pnp = rotation_matrix_to_quaternion(R_pnp)
            
            raw_ang_list.append(ang_pnp)
            raw_rep_list.append(rep_pnp)
            raw_ang_each_frame.append(ang_pnp)
            raw_rep_each_frame.append(rep_pnp)
            if pos_pnp is not None:
                raw_pos_list.append(pos_pnp)
                raw_pos_each_frame.append(pos_pnp)
            else:
                raw_pos_each_frame.append(np.nan)
        else:
            raw_ang_each_frame.append(np.nan)
            raw_pos_each_frame.append(np.nan)
            raw_rep_each_frame.append(np.nan)

       
        start_ekf_time = time.time()
        
        ekf.predict()

        if first_init:
            if r_true is not None and ok:
                if use_trf:
                    ekf.init_state_with_pose(tvec_pnp, phi_pnp)
                else:
                    ekf.init_state(r_true, q_true)
            first_init = False
        else:
            ekf.update(k3d_sel, k2d_sel, camera.cameraMatrix)

        e_r, e_q = ekf.get_pose()
        
        ekf_time = time.time() - start_ekf_time
        timing_info["ekf_times"].append(ekf_time)
        
        total_time = pnp_time + ekf_time
        timing_info["total_times"].append(total_time)
        
        filtered_pose = {
            "time": time_stamp,
            "r_Vo2To_vbs_true": r_true.tolist() if r_true is not None else None,
            "q_vbs2tango_true": q_true.tolist(),
            "r_Vo2To_vbs_pred": e_r.tolist(),
            "q_vbs2tango_pred": e_q.tolist(),
            "filename": fn
        }
        output_poses.append(filtered_pose)
        
        R_pred = quat_to_matrix(e_q)
        R_true = quat_to_matrix(q_true)
        ang_ekf = rotation_angle_error(R_pred, R_true)
        ekf_ang_each_frame.append(ang_ekf)
        ekf_ang_list.append(ang_ekf)

        pos_ekf = None
        if r_true is not None:
            pos_ekf = relative_position_error(e_r, r_true)
            if pos_ekf is not None:
                ekf_pos_list.append(pos_ekf)
            ekf_pos_each_frame.append(pos_ekf if pos_ekf is not None else np.nan)
        else:
            ekf_pos_each_frame.append(np.nan)

        proj_ekf = project_points(
            k3d_sel,
            xyzPhi_to_matrix(e_r[0], e_r[1], e_r[2],
                             ekf.x[3,0], ekf.x[4,0], ekf.x[5,0]),
            camera.cameraMatrix
        )
        diff = proj_ekf - k2d_sel
        rep_ekf = float(np.mean(np.sqrt(np.sum(diff**2, axis=1))))
        ekf_rep_list.append(rep_ekf)
        ekf_rep_each_frame.append(rep_ekf)

    def m_or_none(arr):
        return np.mean(arr) if len(arr) > 0 else None

    raw_ang = m_or_none(raw_ang_list)
    raw_pos = m_or_none(raw_pos_list)
    raw_rep = m_or_none(raw_rep_list)
    ekf_ang = m_or_none(ekf_ang_list)
    ekf_pos = m_or_none(ekf_pos_list)
    ekf_rep = m_or_none(ekf_rep_list)
    
    if timing_info["total_times"]:
        avg_pnp_time = np.mean(timing_info["pnp_times"])
        avg_ekf_time = np.mean(timing_info["ekf_times"])
        avg_total_time = np.mean(timing_info["total_times"])
        
        timing_info["avg_pnp_time"] = avg_pnp_time
        timing_info["avg_ekf_time"] = avg_ekf_time
        timing_info["avg_total_time"] = avg_total_time
        
        timing_info["pnp_fps"] = 1.0 / avg_pnp_time if avg_pnp_time > 0 else 0
        timing_info["ekf_fps"] = 1.0 / avg_ekf_time if avg_ekf_time > 0 else 0
        timing_info["total_fps"] = 1.0 / avg_total_time if avg_total_time > 0 else 0

    return (
        raw_ang, raw_pos, raw_rep,
        ekf_ang, ekf_pos, ekf_rep,
        raw_ang_each_frame, raw_pos_each_frame, raw_rep_each_frame,
        ekf_ang_each_frame, ekf_pos_each_frame, ekf_rep_each_frame,
        output_poses,
        timing_info
    )

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-yolo", action="store_true")
    parser.add_argument("--use-trf", action="store_true", help="是否使用TRF优化位姿估计")
    parser.add_argument("--max-nfev", type=int, default=100, help="TRF优化的最大函数评估次数")
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
        # 如需实时推理，则取消此 else，并加载模型推理
        from ultralytics import YOLO
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
    q_candidates = [0.1]  # 这里可添加更多候选值
    r_candidates = [0.1]  # 同理

    best_score = 1e9
    best_params = None
    best_result = None

    for q_s in q_candidates:
        for r_s in r_candidates:
            (raw_ang, raw_pos, raw_rep,
             ekf_ang, ekf_pos, ekf_rep,
             raw_ang_each_frame, raw_pos_each_frame, raw_rep_each_frame,
             ekf_ang_each_frame, ekf_pos_each_frame, ekf_rep_each_frame,
             output_poses,
             timing_info
            ) = run_6dof_ekf_pipeline_cv_lie(
                camera, poses_data, kpts3d_obj, yolo_results,
                q_scale=q_s, r_scale=r_s, use_trf=args.use_trf, max_nfev=args.max_nfev,
                min_kpt_required=6, dt=1/30
            )

            # 若EKF结果全是 None，跳过
            if ekf_ang is None or ekf_pos is None or ekf_rep is None:
                continue

            # 简易 score
            score = np.radians(ekf_ang) + ekf_pos
            if score < best_score:
                best_score = score
                best_params = (q_s, r_s)
                best_result = (
                    raw_ang, raw_pos, raw_rep,
                    ekf_ang, ekf_pos, ekf_rep,
                    raw_ang_each_frame, raw_pos_each_frame, raw_rep_each_frame,
                    ekf_ang_each_frame, ekf_pos_each_frame, ekf_rep_each_frame,
                    output_poses,
                    timing_info
                )

            print(f"[*] q={q_s}, r={r_s}, raw_ang={raw_ang}, ekf_ang={ekf_ang}, score={score:.3f}")

    if best_params is None:
        print("[WARN] no valid params found.")
        return

    q_s, r_s = best_params
    (raw_ang, raw_pos, raw_rep,
     ekf_ang, ekf_pos, ekf_rep,
     raw_ang_each_frame, raw_pos_each_frame, raw_rep_each_frame,
     ekf_ang_each_frame, ekf_pos_each_frame, ekf_rep_each_frame,
     output_poses,
     timing_info
    ) = best_result

    print("\n========= 最优参数 (CV + Lie) =========")
    print(f"q_scale={q_s}, r_scale={r_s}")
    print("\n========= 对比结果 (CV + Lie) =========")
    if args.use_trf:
        print(f"[已启用TRF优化，max_nfev={args.max_nfev}]")
    print("[未滤波 solvePnP]:")
    print(f" 角度误差={raw_ang}, 相对位置误差={raw_pos}, 重投影误差={raw_rep}")
    raw_score = np.radians(raw_ang) + raw_pos
    print(f" 综合Score={raw_score:.3f}")
    print("[6DoF EKF (Lie)]:")
    print(f" 角度误差={ekf_ang}, 相对位置误差={ekf_pos}, 重投影误差={ekf_rep}")
    print(f" 综合Score={best_score:.3f}")
    
    # 打印性能信息
    print("\n========= 性能指标 =========")
    print(f"PnP 平均处理时间: {timing_info['avg_pnp_time']*1000:.2f} ms, FPS: {timing_info['pnp_fps']:.2f}")
    print(f"EKF 平均处理时间: {timing_info['avg_ekf_time']*1000:.2f} ms, FPS: {timing_info['ekf_fps']:.2f}")
    print(f"总平均处理时间: {timing_info['avg_total_time']*1000:.2f} ms, FPS: {timing_info['total_fps']:.2f}")

    # 创建输出目录
    output_dir = f"blender_proc_scripts/kalmancmp/cmpimg/{casename}"
    os.makedirs(output_dir, exist_ok=True)
    
    # 图片输出路径
    save_path = f"{output_dir}/{casename}kalman_kptobs_EKF_lie.png"
        
    # 保存指标结果到txt文件
    txt_save_path = f"{output_dir}/{casename}kalman_kptobs_EKF_lie.txt"
    with open(txt_save_path, 'w') as f:
        f.write(f"Raw Score={raw_score:.4f}\n")
        f.write(f"Filtered Score={best_score:.4f}\n")
        f.write(f"Raw angle error mean: {np.mean(raw_ang_each_frame):.4f}\n")
        f.write(f"Raw radians angle error mean: {np.mean(np.radians(raw_ang_each_frame)):.4f}\n")
        f.write(f"Filtered angle error mean: {np.mean(ekf_ang_each_frame):.4f}\n") 
        f.write(f"Filtered radians angle error mean: {np.mean(np.radians(ekf_ang_each_frame)):.4f}\n") 
        f.write(f"Raw position error mean: {np.mean(raw_pos_each_frame):.4f}\n")
        f.write(f"Filtered  position error mean: {np.mean(ekf_pos_each_frame):.4f}\n")
    
    # 保存JSON输出
    json_save_path = f"{output_dir}/{casename}kalman_kptobs_EKF_lie.json"
    
    # 保存结果（使用简单的原始格式）
    with open(json_save_path, 'w') as f:
        json.dump(output_poses, f, indent=4)
    
    print(f"JSON结果已保存到: {json_save_path}")
    
    # -------- 绘图：逐帧误差对比 (Angle / Position / Reproj) ---------
    frame_nums = np.arange(len(raw_ang_each_frame))

    plt.figure(figsize=(15,5))

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
    plt.tick_params(axis='both', which='major', labelsize=tick_size)

    # 子图2：相对位置误差
    plt.subplot(1,2,2)
    plt.plot(frame_nums, raw_pos_each_frame, linestyle='-', marker='o', color='red', linewidth=1.5, label='未滤波 相对位置误差')
    plt.plot(frame_nums, ekf_pos_each_frame, linestyle='--', marker='x', color='blue', linewidth=1.5, label='滤波后 相对位置误差')
    plt.xlabel('帧编号', fontsize=label_size)
    plt.ylabel('相对位置误差', fontsize=label_size)
    plt.title('相对位置误差随帧变化', fontsize=title_size)
    plt.legend(fontsize=legend_size)
    plt.grid(True)
    plt.tick_params(axis='both', which='major', labelsize=tick_size)

    plt.tight_layout()
    # 调整图形大小以匹配屏幕分辨率
    fig = plt.gcf()
    fig.set_size_inches(25.6, 16)  # 基于2560x1600的分辨率
    # 保存图像

    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    # plt.show()
    
    print(f"图像结果已保存到: {save_path}")
    print(f"文本结果已保存到: {txt_save_path}")

if __name__ == "__main__":
    main()