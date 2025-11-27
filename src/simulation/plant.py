import numpy as np
import yaml
from scipy.integrate import solve_ivp

# from src.simulation.vehicle_model import VehicleModel, compile_vehicle_model
# from src.simulation.bump import Bump


class Vehicle_Parameters:
    def __init__(self, **kwargs):
        # Default parameters
        self.T = 10.0          # Default Total simulation time
        self.time_step = 0.01  # Default Time step
        
        self.k_s_f = 35000.0  # Front suspension stiffness [N/m]
        self.k_s_r = 30000.0  # Rear suspension stiffness [N/m]
        self.c_s_f = 2500.0   # Front suspension damping [N*s/m]
        self.c_s_r = 2000.0   # Rear suspension damping [N*s/m]
        self.k_us_f = 250000.0 # Front unsprung stiffness [N/m]
        self.k_us_r = 200000.0 # Rear unsprung stiffness [N/m]
        self.l_f = 1.1        # Distance from CG to front axle [m]
        self.l_r = 1.6        # Distance from CG to rear axle [m]
        self.C_r_f = 0.015    # Rolling resistance coefficient front
        self.C_r_r = 0.015    # Rolling resistance coefficient rear
        self.m_tot = 1500.0   # Total vehicle mass [kg]
        self.m_us_f = 50.0    # Front unsprung mass [kg]
        self.m_us_r = 45.0    # Rear unsprung mass [kg]
        self.I = 2500.0       # Moment of inertia [kg*m^2]
        self.r_wheel = 0.3    # Wheel radius [m]
        self.mu = 0.9         # Friction coefficient
        self.eta_drive = 0.9  # Drivetrain efficiency
        self.r_transmission = 3.5 # Transmission ratio
        self.rho_air = 1.225  # Air density [kg/m^3]
        self.A = 2.2          # Frontal area [m^2]
        self.C_d = 0.3        # Drag coefficient
        self.h_f = 0.0        # Height of front roll center [m]
        self.h_r = 0.0        # Height of rear roll center [m]
        self.phi_f = 0.0      # Front roll steer angle [rad]
        self.phi_r = 0.0      # Rear roll steer angle [rad]

        # Overwrite defaults with any provided kwargs
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)

def create_plant_from_config():
    try:
        with open("configs/simulations.yaml", 'r') as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        print("Warning: simulations.yaml not found. Using default parameters.")
        config = {}
        
    params = Vehicle_Parameters(**config)
    
    # Import here to avoid circular import
    from src.simulation.vehicle_model import compile_vehicle_model
    from src.simulation.bump import Bump
    
    vehicle = compile_vehicle_model(params)
    bump = Bump()
    return Plant(vehicle, bump, params)


class ODESystem:
    def __init__(self, vehicle, bump, params):
        self.vehicle = vehicle
        self.bump = bump
        self.params = params
        self.u = 0.0 # Control input to be set before integration

    def f(self, t, y):
        # y is state vector (10 dim)
        # vehicle model expects: x(10), u(1), z(2)
        
        x_com = y[9] 
        
        theta = y[6] 
        
        l_f = self.params.l_f
        l_r = self.params.l_r
        
        x_f = x_com + l_f * np.cos(theta)
        x_r = x_com - l_r * np.cos(theta)
        
        z_road_f = self.bump(x_f)
        z_road_r = self.bump(x_r)
        
        z_data = np.array([z_road_f, z_road_r])
        
        d_state = self.vehicle(y, self.u, z_data)
        
        return d_state

    def integrate(self, t_span, y0, dt):
        # Ensure t_span is a tuple of (start, end)
        if not isinstance(t_span, tuple) or len(t_span) != 2:
            raise ValueError("t_span must be a tuple of (start, end)")

        # Ensure dt is a positive number
        if dt <= 0:
            raise ValueError("dt must be a positive number")

        sol = solve_ivp(self.f, t_span, y0, method='RK45', max_step=dt, rtol=1e-6, atol=1e-6)
        return sol

class Plant:
    def __init__(self, vehicle, bump, params):
        self.vehicle = vehicle
        self.bump = bump
        self.params = params
        self.ode = ODESystem(vehicle, bump, params)
        self.state = np.zeros(10)
        self.time = 0.0

    def reset(self):
        self.state = np.zeros(10)
        # Initial velocity x_dot = 10.0 m/s (index 4)
        self.state[4] = 10.0
        self.time = 0.0
        return self.state

    def step(self, action, dt=0.01):
        # Set control input in ODE system
        self.ode.u = action
        
        # Update state using ODE solver
        t_span = (self.time, self.time + dt)
        sol = self.ode.integrate(t_span, self.state, dt)
        
        # Update internal state
        self.state = sol.y[:, -1]
        self.time += dt
        
        return self.state
