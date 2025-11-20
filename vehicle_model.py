import numpy as np
from numba import jit
import math

from plant import Vehicle_Parameters

@jit(nopython=True, cache=True)
def _compute_dynamics(x_data, u, z_data, params_data, out_data):
    # Unpack constants
    k_s_f, k_s_r          = params_data[0], params_data[1]
    c_s_f, c_s_r          = params_data[2], params_data[3] 
    k_us_f, k_us_r        = params_data[4], params_data[5]
    l_f, l_r              = params_data[6], params_data[7]
    c_r_f, c_r_r          = params_data[8], params_data[9]
    m_tot, m_us_f, m_us_r = params_data[10], params_data[11], params_data[12]
    I, r_wheel            = params_data[13], params_data[14]
    mu_g, eta_r, k_air    = params_data[15], params_data[16], params_data[17] # mu*g, eta*r_trans, k_air
    gravity_half          = params_data[18]
    h_f, h_r              = params_data[19], params_data[20]
    tan_phi_f, tan_phi_r  = params_data[21], params_data[22]

    # state variables
    dz_com, dtheta, dz_us_f, dz_us_r, dx_com = x_data[0:5]
    z_com, theta, z_us_f, z_us_r, x_com      = x_data[5:10]

    # Precompute
    sin_theta = math.sin(theta)
    cos_theta = math.cos(theta)
    sign_dx_com = 1.0 if dx_com > 0 else (-1.0 if dx_com < 0 else 0.0)

    # Suspension
    del_f = l_f * sin_theta + z_com - z_us_f
    del_r = -l_r * sin_theta + z_com - z_us_r
    ddel_f = l_f * dtheta * cos_theta + dz_com - dz_us_f
    ddel_r = -l_r * dtheta * cos_theta + dz_com - dz_us_r

    spring_force_f = k_s_f * del_f
    spring_force_r = k_s_r * del_r
    damper_force_f = c_s_f * ddel_f
    damper_force_r = c_s_r * ddel_r

    # Unsprung force
    unsprung_force_f = k_us_f * (z_us_f - z_data[0])
    unsprung_force_r = k_us_r * (z_us_r - z_data[1])

    # Drivetrain
    wheel_torque = eta_r * u
    traction_force = wheel_torque / r_wheel

    # Clamp traction force
    F_max = mu_g
    if traction_force > F_max:
        traction_force = F_max
    elif traction_force < -F_max:
        traction_force = -F_max

    # Aerodynamic and rolling resistance
    air_drag = k_air * dx_com * dx_com * sign_dx_com
    F0 = gravity_half * sign_dx_com
    rolling_f = c_r_f * F0
    rolling_r = c_r_r * F0

    # Tire forces
    tire_long_f = -rolling_f
    tire_long_r = traction_force - rolling_r
    tire_vert_f = -tire_long_f * tan_phi_f
    tire_vert_r = tire_long_r * tan_phi_r

    # Pitch moment
    pitch_moment_extra = (tire_long_f * tan_phi_f * l_f + 
                         tire_long_r * tan_phi_r * l_r)
    # pitch_moment_extra = (tire_long_f * tan_phi_f * l_f + tire_long_r * tan_phi_r * l_r) * cos_theta

    # Compute accelerations
    out_data[0] = (-(spring_force_f + damper_force_f + spring_force_r + damper_force_r) + 
                   (tire_vert_f + tire_vert_r)) / m_tot

    out_data[1] = (-(spring_force_f + damper_force_f) * l_f * cos_theta + 
                   (spring_force_r + damper_force_r) * l_r * cos_theta + 
                   traction_force * h_r - rolling_f * h_f - rolling_r * h_r + 
                   pitch_moment_extra) / I

    out_data[2] = (spring_force_f + damper_force_f - unsprung_force_f - tire_vert_f) / m_us_f
    out_data[3] = (spring_force_r + damper_force_r - unsprung_force_r - tire_vert_r) / m_us_r
    out_data[4] = (traction_force - air_drag - rolling_f - rolling_r) / m_tot

    # velocities
    out_data[5] = dz_com
    out_data[6] = dtheta
    out_data[7] = dz_us_f  
    out_data[8] = dz_us_r
    out_data[9] = dx_com


class VehicleModel:
    __slots__ = ('_params_arr', '_output_buffer', '_x_buffer', '_z_buffer')
    def __init__(self, params=None):
        if params is None:
            params = Vehicle_Parameters()
        self._params_arr = np.array([
            params.k_s_f, params.k_s_r,                    # 0, 1
            params.c_s_f, params.c_s_r,                    # 2, 3
            params.k_us_f, params.k_us_r,                  # 4, 5
            params.l_f, params.l_r,                        # 6, 7
            params.C_r_f, params.C_r_r,                    # 8, 9
            params.m_tot, params.m_us_f, params.m_us_r,    # 10, 11, 12
            params.I, params.r_wheel,                      # 13, 14
            params.mu * params.m_tot * 9.81,               # 15 (F_max)
            params.eta_drive * params.r_transmission,      # 16 (eta * r_trans)
            0.5 * params.rho_air * params.A * params.C_d,  # 17 (k_air)
            params.m_tot * 9.81 / 2,                       # 18 (gravity_half)
            params.h_f, params.h_r,                        # 19, 20
            np.tan(params.phi_f), np.tan(params.phi_r)     # 21, 22
        ], dtype=np.float64)

        # buffers
        self._output_buffer = np.zeros(10, dtype=np.float64)
        self._x_buffer = np.zeros(10, dtype=np.float64) 
        self._z_buffer = np.zeros(2, dtype=np.float64)

    def __call__(self, x, u, z):
        return self.system_dynamics(x, u, z)

    def system_dynamics(self, x, u, z):
        self._x_buffer[:] = x
        self._z_buffer[:] = z

        # Call JIT-compiled function
        _compute_dynamics(self._x_buffer, float(u), self._z_buffer, 
                         self._params_arr, self._output_buffer)
        return self._output_buffer

    def get_K_matrix(self, x0, ueq, zeq, Q, R):
        import scipy.linalg
        A, B = self.get_ABmatrix(x0, ueq, zeq)

        P = scipy.linalg.solve_continuous_are(A, B, Q, R)
        K = np.linalg.inv(R) @ B.T @ P

        return K

    def get_ABmatrix(self, xeq, ueq, zeq):
        def f(x,u):
            dX = self.system_dynamics(x, u, zeq)
            return np.array(dX)

        n = 10; m = 1; eps = 1e-1

        # A 
        A = np.zeros((n,n))
        for i in range(n):
            dx = np.zeros(n); dx[i] = eps
            A[:,i] = (f(xeq+dx, ueq) - f(xeq-dx, ueq)) / (2*eps)

        # B
        B = np.zeros((n,m))
        for j in range(m):
            du = np.zeros(m); du[j] = eps
            B[:,j] = (f(xeq, ueq+du) - f(xeq, ueq-du)) / (2*eps)
        return A, B

    def check_control_validity(self, u):
        wheel_torque = self._params_arr[16] * u  # eta_r * u
        traction_force = wheel_torque / self._params_arr[14]  # T_w / r_wheel
        
        F_max = self._params_arr[15]  # mu * m_tot * g
        if abs(traction_force) > F_max:
            print("Warning: Control input exceeds traction limits.")
            return False
        
        if abs(traction_force) / self._params_arr[10] > 0.2 * 9.81:  # F_T_r / m_tot > 0.2g
            print("Warning: Control input exceeds acceleration limits.")
            return False
        
        return True

# Warm-up compilation function
def compile_vehicle_model(params=None):
    model = VehicleModel(params=params)
    test_state = np.zeros(10)
    test_state[4] = 10.0  # velocity
    test_control = 0.0
    test_road = np.zeros(2)

    # Trigger JIT compilation
    model(test_state, test_control, test_road)
    return model