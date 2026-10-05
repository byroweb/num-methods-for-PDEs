"""
Actual setup on 5 Oct: PC off, north blinds closed, AC setpoint raised from 70 F to 74 F at 06:00.

Two regimes:
  free float  (room < 74 F): the linear Laplace model applies unchanged; the AC switches on
               when theta_a(t) first reaches the setpoint -> found from the Laplace solution.
  thermostat  (room = 74 F): the room temperature is pinned, the AC removes whatever the
               envelope, slab, skylight and dogs deliver.  The switch makes the overall system
               nonlinear, so the load is computed with the finite-difference model plus an
               ideal (stiff proportional) thermostat, checked against Laplace before the switch.
"""
import numpy as np
from scipy.integrate import solve_ivp
from apartment_heat import (params, solve_laplace, diffuse_kinks, T_OUT_KINKS, T0_F, F2C, HR, c2f)

T_SET_F = 74.0
G_AC = 2.0e4          # W/K thermostat gain (offset ~0.06 K at full load)

def fd_thermostat(p, t_end, theta_set, N=40):
    dz = p["d"]/N
    To = lambda t: sum(a*max(t - tk, 0) for tk, a in T_OUT_KINKS)
    dk = diffuse_kinks(p)
    Ri = p["R_fin"] + 1/p["h_in"]
    def parts(t, y):
        T, Ta = y[:N], y[N]
        Q = p["Q_dogs"] + sum(a*max(t - tk, 0) for tk, a in dk)
        Q_ac = -G_AC*max(0.0, Ta - theta_set)
        q_bot = (To(t) - T[0])/(1/p["h_g"] + dz/(2*p["k"]))
        q_top = (T[-1] - Ta)/(Ri + dz/(2*p["k"]))
        return T, Ta, Q, Q_ac, q_bot, q_top
    def rhs(t, y):
        T, Ta, Q, Q_ac, q_bot, q_top = parts(t, y)
        lap = np.zeros(N)
        lap[1:-1] = T[2:] - 2*T[1:-1] + T[:-2]
        lap[0] = T[1] - T[0]
        lap[-1] = T[-2] - T[-1]
        dT = p["k"]*lap/dz**2
        dT[0] += q_bot/dz
        dT[-1] -= q_top/dz
        dTa = (p["A_f"]*q_top + p["K_env"]*(To(t) - Ta) + Q + Q_ac)/p["C_a"]
        return np.concatenate([dT/p["rhoc"], [dTa]])
    q0 = -p["theta_a0"]/p["R_tot"]
    z = (np.arange(N) + 0.5)*dz
    y0 = np.concatenate([-q0*(1/p["h_g"] + z/p["k"]), [p["theta_a0"]]])
    t = np.linspace(0, t_end, 541)
    sol = solve_ivp(rhs, (0, t_end), y0, t_eval=t, method="BDF", max_step=120, rtol=1e-6, atol=1e-8)
    Ta = sol.y[N]
    Q_ac = np.array([-G_AC*max(0.0, x - theta_set) for x in Ta])
    q_top = np.array([(sol.y[N - 1, i] - Ta[i])/(Ri + dz/(2*p["k"])) for i in range(len(t))])
    q_bot = np.array([(To(tt) - sol.y[0, i])/(1/p["h_g"] + dz/(2*p["k"])) for i, tt in enumerate(t)])
    T_floor = Ta + q_top/p["h_in"]                 # floor surface, room side
    return t, Ta, Q_ac, T_floor, q_top, q_bot, np.array([To(tt) for tt in t])

if __name__ == "__main__":
    p = params()
    p["blind"] = 0.3
    p["Q_dogs"] = 100.0                       # PC off: dogs only
    theta_set = F2C*(T_SET_F - T0_F)
    # free-float Laplace solution -> switch-on time
    tl = np.linspace(0, 9*HR, 541)
    r = solve_laplace(p, tl)
    Ta_free = r["Th_a"] - r["Th_a_court"]     # courtyard ~0 (not adjacent)
    i_on = int(np.argmax(Ta_free >= theta_set))
    t_on = np.interp(theta_set, Ta_free[i_on-1:i_on+1], tl[i_on-1:i_on+1])
    # thermostat run
    t, Ta, Q_ac, T_floor, q_top, q_bot, To = fd_thermostat(p, 9*HR, theta_set)
    i_fd = int(np.argmax(Q_ac < -1.0))
    F = lambda x: T0_F + c2f(x)
    print(f"Laplace: room reaches {T_SET_F} F at {6 + t_on/HR:.2f} h  "
          f"({int(6 + t_on/HR)}:{int(60*((t_on/HR) % 1)):02d})")
    print(f"FD     : AC first runs at {6 + t[i_fd]/HR:.2f} h;  free-float max diff before switch "
          f"{np.max(abs(np.interp(t[:i_fd], tl, Ta_free) - Ta[:i_fd]))*9/5:.3f} F")
    E_th = -np.trapezoid(Q_ac, t)/3.6e6
    print(f"AC heat removed 06-15: {E_th:.2f} kWh thermal  (~{E_th/3.0:.2f} kWh electric at COP 3)")
    print(f"free-float at 15:00 would be {F(Ta_free[-1]):.1f} F")
    print(" clock  T_out  T_room  T_floor  AC_W   BTU/h  floor_into_room_W  garage_into_slab_W")
    for h in range(6, 16):
        i = int(np.argmin(abs(6 + t/HR - h)))
        print(f" {h:5d} {F(To[i]):6.1f} {F(Ta[i]):7.1f} {F(T_floor[i]):8.1f} {-Q_ac[i]:6.0f} "
              f"{-Q_ac[i]*3.412:7.0f} {p['A_f']*q_top[i]:9.0f} {p['A_f']*q_bot[i]:9.0f}")
    np.savez("thermostat_74.npz", t=t, Ta=Ta, Q_ac=Q_ac, T_floor=T_floor, To=To, Ta_free=np.interp(t, tl, Ta_free))
