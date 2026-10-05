"""
Laplace-transform model of an unoccupied (2 dogs) San Diego apartment, 5 Oct 2026, 06:00-15:00.

Unknowns are deviations from a uniform initial state T0 (room, slab, outdoors all at the
06:00 outdoor temperature), so every initial condition is zero and the Laplace transform of
d/dt is simply s.

Heat routes
  1  garage air -> 8" concrete floor slab -> room            (distributed 1-D slab, exact in s)
  2  sun step on a courtyard slab -> lateral conduction in the podium slab -> unit
  3  outdoor air -> N/E walls, double glazing, infiltration (wind dependent)
  3b diffuse daylight through the N/E glazing (no direct sun, but skylight is not zero)
  +  two dogs (constant internal gain)

Inversion: Talbot contour (mpmath) for the *undelayed* unit-ramp/step responses; delayed
inputs are assembled with the second shift theorem  L^-1{e^{-a s}F(s)} = f(t-a)H(t-a).
"""
import json
import numpy as np
import mpmath as mp
import sympy as sp
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import solar

FT = 0.3048
HR = 3600.0
F2C = 5/9                      # temperature *difference* conversion

def c2f(dT):                   # temperature difference K -> degF
    return dT*9/5

# ----------------------------------------------------------------------------------------
# Weather (forecast for San Diego, 5 Oct 2026; Extreme Heat Warning in effect)
# ----------------------------------------------------------------------------------------
FORECAST_F = [(6, 77), (7, 79), (8, 80), (9, 82), (10, 88), (11, 97),
              (12, 98), (13, 98), (14, 99), (15, 99)]
T0_F = FORECAST_F[0][1]
WIND_10M = {"calm": 1.0, "forecast": 3.4, "breezy": 4.5}   # m/s  (forecast W 5-10 mph)

# ----------------------------------------------------------------------------------------
# Building parameters (assumptions are flagged in the write-up)
# ----------------------------------------------------------------------------------------
def params(v10=WIND_10M["forecast"], T_room0_F=70.0, east_glass_ft2=0.0):
    p = {}
    p["A_f"] = 30*25*FT**2                       # floor slab under unit, m^2
    p["d"], p["k"], p["rhoc"] = 0.20, 1.6, 2300*900.0   # 8" concrete
    p["alpha"] = p["k"]/p["rhoc"]
    # underside (garage): shaded, open; wind inside ~30% of 10 m wind
    v_g = 0.3*v10
    p["h_g"] = (1.5 + 3.0*v_g) + 5.7              # convection + radiation to garage surfaces
    # room side of floor: still air + radiation to room surfaces, thin vinyl finish
    p["h_in"], p["R_fin"] = 7.5, 0.02
    # envelope: north 30' x 12', east 25' x 12'; two 9' x 4' north windows (only lit glass)
    A_wall_tot = (30 + 25)*12*FT**2
    p["A_glass_N"] = 2*9*4*FT**2
    p["A_glass"] = p["A_glass_N"] + east_glass_ft2*FT**2   # east glass (if any) sees no daylight
    p["A_opq"] = A_wall_tot - p["A_glass"]
    v_loc = 0.4*v10                               # N/E faces are leeward of a W wind
    p["h_o"] = 5.7 + 3.8*v_loc                    # McAdams-type exterior film
    p["U_opq"] = 1/(13*0.176 + 0.12 + 1/p["h_o"])   # R-13 effective framed wall
    p["U_glass"] = 1/(0.19 + 0.13 + 1/p["h_o"])     # double glazed, Al frame (~2.8 nominal)
    V = p["A_f"]*12*FT
    p["ACH"] = 0.20 + 0.05*v10                    # wind-driven infiltration, closed unit
    p["UA_inf"] = 1.18*1006*V*p["ACH"]/HR
    p["UA_opq"] = p["U_opq"]*p["A_opq"]
    p["UA_glass"] = p["U_glass"]*p["A_glass"]
    p["K_env"] = p["UA_opq"] + p["UA_glass"] + p["UA_inf"]
    # room air + furniture + drywall effective capacitance
    p["C_a"] = 2.5e6
    # internal / diffuse gains
    p["Q_dogs"] = 100.0                           # two ~25 kg dogs, sensible part
    p["SHGC_dif"] = 0.50                          # double glazing, hemispherical (diffuse) SHGC
    p["f_vert"] = 0.12      # I_north ~ 0.5*DHI + 0.5*rho_g*GHI ~ (0.5*0.14 + 0.5*0.10)*GHI
    # AC holds the room at T_room0 until 06:00 (steady state); garage/outdoor at T0
    p["theta_a0"] = F2C*(T_room0_F - T0_F)
    p["R_tot"] = 1/p["h_g"] + p["d"]/p["k"] + p["R_fin"] + 1/p["h_in"]
    p["GQ0"] = 1/(p["A_f"]/p["R_tot"] + p["K_env"])   # G_Q(0), K per W
    p["Q_pre"] = p["theta_a0"]/p["GQ0"]               # dogs + AC before 06:00 (negative = cooling)
    # route 2: courtyard slab
    p["alpha_s"] = 0.65
    p["t_sun_clock"] = 11.0                       # sun clears the building (assumed)
    p["h_top"] = 5.7 + 3.8*(0.5*v10)              # exposed slab top surface
    p["edge"] = 15*FT                             # shared edge length if flush (worst case)
    p["v10"] = v10
    return p

# ----------------------------------------------------------------------------------------
# Inputs as sums of shifted ramps / steps  (exact Laplace transforms)
# ----------------------------------------------------------------------------------------
def ramp_kinks(knots_clock, values):
    """Piecewise-linear f(t), f(0)=0, as sum_k dm_k (t-t_k) H(t-t_k).
    L{f} = sum_k dm_k e^{-t_k s}/s^2.  Returns [(t_k [s], dm_k [unit/s])]."""
    t = (np.asarray(knots_clock, float) - knots_clock[0])*HR
    v = np.asarray(values, float) - values[0]
    slopes = np.diff(v)/np.diff(t)
    out, prev = [], 0.0
    for tk, m in zip(t[:-1], slopes):
        out.append((tk, m - prev))
        prev = m
    out.append((t[-1], -prev))        # hold constant after last knot
    return out

T_OUT_KINKS = ramp_kinks([h for h, _ in FORECAST_F], [F2C*(f - T0_F) for _, f in FORECAST_F])

DIF_KNOTS = [6.0, 6.85] + list(range(7, 16))

def diffuse_kinks(p):
    """Skylight on the north glass (+ small sol-air bump on opaque walls), following the
    clear-sky curve hour by hour -> again a sum of shifted ramps."""
    I = p["f_vert"]*solar.ghi_clear(np.array(DIF_KNOTS, float))
    I[:2] = 0.0
    q = I*(p["SHGC_dif"]*p["A_glass_N"] + p["UA_opq"]*0.6/p["h_o"])
    return ramp_kinks(DIF_KNOTS, q)

# ----------------------------------------------------------------------------------------
# Transfer functions (functions of complex s, mpmath)
# ----------------------------------------------------------------------------------------
def slab_ABCD(p, s):
    """Thermal quadrupole: [T; q]_garage-air = M [T; q]_room-air, q = upward flux W/m^2."""
    g = mp.sqrt(s/p["alpha"])
    gd = g*p["d"]
    S = mp.matrix([[mp.cosh(gd), mp.sinh(gd)/(p["k"]*g)],
                   [p["k"]*g*mp.sinh(gd), mp.cosh(gd)]])
    Fg = mp.matrix([[1, 1/p["h_g"]], [0, 1]])
    Fi = mp.matrix([[1, p["R_fin"] + 1/p["h_in"]], [0, 1]])
    return Fg*S*Fi

def G_room(p, s):
    """Theta_a = G_o(s) Theta_o + G_Q(s) Q."""
    M = slab_ABCD(p, s)
    A, B = M[0, 0], M[0, 1]
    den = p["C_a"]*s + p["A_f"]*A/B + p["K_env"]
    return (p["A_f"]/B + p["K_env"])/den, 1/den

def floor_flux(p, s, Th_o, Th_a):
    M = slab_ABCD(p, s)
    return (Th_o - M[0, 0]*Th_a)/M[0, 1]          # W/m^2 into room

def courtyard_edge_flux(p, s, L):
    """Route 2.  Fin model of the podium slab (lumped through thickness -> upper bound on
    lateral spread).  x<0 sunlit courtyard (semi-infinite), x>0 shaded interior slab.
    Returns flux into the unit at x=L for a UNIT step of absorbed sun (W per m of edge per W/m^2)."""
    kd = p["k"]*p["d"]
    m1 = mp.sqrt(s/p["alpha"] + (p["h_top"] + p["h_g"])/kd)
    m2 = mp.sqrt(s/p["alpha"] + (1/(p["R_fin"] + 1/p["h_in"]) + p["h_g"])/kd)
    P = 1/(s*kd)
    return kd*m2*P*mp.exp(-m2*L)/(m1*(m1 + m2))

# ----------------------------------------------------------------------------------------
# Numerical inversion
# ----------------------------------------------------------------------------------------
mp.mp.dps = 20

def invert(F, tgrid):
    out = np.zeros_like(tgrid)
    for i, t in enumerate(tgrid):
        if t > 0:
            out[i] = float(mp.invertlaplace(F, t, method="talbot"))
    return out

def shift_sum(base_t, base_f, kinks, t):
    """sum_k a_k * f(t - t_k) H(t - t_k), f known on base_t (second shift theorem)."""
    f = interp1d(base_t, base_f, bounds_error=False, fill_value=(0.0, base_f[-1]))
    return sum(a*np.where(t > tk, f(t - tk), 0.0) for tk, a in kinks)

def ac_off(p, unit_step, dc=None):
    """Initial condition by superposition.  Before 06:00 the heat input to the room is the
    constant Q_pre = dogs + AC, which holds the whole slab/room system in steady state at
    theta_a0.  At t=0 the AC stops, i.e. Q jumps from Q_pre to Q_dogs:
        Theta = G(0) Q_pre / s  +  (Q_dogs - Q_pre) G(s)/s
    No initial temperature profile in the slab is ever needed."""
    dc = p["GQ0"] if dc is None else dc
    return dc*p["Q_pre"] + (p["Q_dogs"] - p["Q_pre"])*unit_step

def solve_laplace(p, t, L_court=0.0):
    tb = np.linspace(0, t[-1], 181)
    # unit-ramp responses (1/s^2) and unit-step responses (1/s)
    r_o = invert(lambda s: G_room(p, s)[0]/s**2, tb)          # room to outdoor ramp
    r_Q = invert(lambda s: G_room(p, s)[1]/s**2, tb)          # room to heat ramp
    u_Q = invert(lambda s: G_room(p, s)[1]/s, tb)             # room to heat step
    # floor flux responses
    def flux_ramp(s):
        Go, _ = G_room(p, s)
        return floor_flux(p, s, 1/s**2, Go/s**2)
    def flux_Qramp(s):
        _, GQ = G_room(p, s)
        return floor_flux(p, s, 0, GQ/s**2)
    def flux_Qstep(s):
        _, GQ = G_room(p, s)
        return floor_flux(p, s, 0, GQ/s)
    fr_o, fr_Q, fu_Q = invert(flux_ramp, tb), invert(flux_Qramp, tb), invert(flux_Qstep, tb)

    # route 2: courtyard -> edge flux (W per W/m^2 absorbed), then into room as heat
    t_sun = (p["t_sun_clock"] - 6)*HR
    q_sun = p["alpha_s"]*0.85*float(np.mean(solar.ghi_clear(np.linspace(p["t_sun_clock"], 15, 20))))
    ce = invert(lambda s: courtyard_edge_flux(p, s, L_court), tb)*p["edge"]*q_sun
    Q_court = shift_sum(tb, ce, [(t_sun, 1.0)], t)
    # room response to the courtyard heat (Duhamel on the step response, discrete)
    dQc = np.diff(np.concatenate([[0], interp1d(t, Q_court)(tb)]))
    resp_c = sum(dq*np.where(tb >= tb[i], np.interp(tb - tb[i], tb, u_Q), 0) for i, dq in enumerate(dQc))

    dk = diffuse_kinks(p)
    Th_o = shift_sum(tb, tb, T_OUT_KINKS, t)
    Th_a = (shift_sum(tb, r_o, T_OUT_KINKS, t) + shift_sum(tb, r_Q, dk, t)
            + ac_off(p, np.interp(t, tb, u_Q)) + np.interp(t, tb, resp_c))
    q_floor = (shift_sum(tb, fr_o, T_OUT_KINKS, t) + shift_sum(tb, fr_Q, dk, t)
               + ac_off(p, np.interp(t, tb, fu_Q), dc=-p["GQ0"]/p["R_tot"]))
    Q_dif = shift_sum(tb, tb, dk, t)
    Th_a_dif = shift_sum(tb, r_Q, dk, t)                      # superposition pieces
    Th_a_court = np.interp(t, tb, resp_c)
    q_floor_dif = shift_sum(tb, fr_Q, dk, t)
    return dict(t=t, Th_o=Th_o, Th_a=Th_a, q_floor=q_floor, Q_dif=Q_dif, Q_court=Q_court,
                Th_a_dif=Th_a_dif, Th_a_court=Th_a_court, q_floor_dif=q_floor_dif,
                q_sun=q_sun)

# ----------------------------------------------------------------------------------------
# Lumped 2-node model: closed-form partial fractions (sympy)
# ----------------------------------------------------------------------------------------
def lumped_symbolic(p):
    """2-node RC (slab node + room node).  G(s) = N(s)/D(s) rational, so every response is a
    finite sum of residues:  L^-1{G/s^2} = G(0) t + G'(0) + sum_i r_i e^{p_i t}/p_i^2,
    L^-1{G/s} = G(0) + sum_i r_i e^{p_i t}/p_i,  r_i = N(p_i)/D'(p_i)."""
    s = sp.symbols("s")
    Cs = p["rhoc"]*p["d"]*p["A_f"]
    Rg = 1/(p["h_g"]*p["A_f"]) + p["d"]/(2*p["k"]*p["A_f"])
    Ri = p["d"]/(2*p["k"]*p["A_f"]) + (p["R_fin"] + 1/p["h_in"])/p["A_f"]
    Ca, K = p["C_a"], p["K_env"]
    Msys = sp.Matrix([[Cs*s + 1/Rg + 1/Ri, -1/Ri], [-1/Ri, Ca*s + 1/Ri + K]])
    D = sp.expand(Msys.det())
    No = sp.expand((Cs*s + 1/Rg + 1/Ri)*K + (1/Ri)*(1/Rg))   # Cramer, room row, outdoor input
    NQ = sp.expand(Cs*s + 1/Rg + 1/Ri)                         # Cramer, room row, heat input
    Dp = np.array(sp.Poly(D, s).all_coeffs(), float)
    poles = np.roots(Dp)
    def build(N):
        Np = np.array(sp.Poly(N, s).all_coeffs(), float)
        G0 = np.polyval(Np, 0)/np.polyval(Dp, 0)
        dG0 = (np.polyval(np.polyder(Np), 0)*np.polyval(Dp, 0)
               - np.polyval(Np, 0)*np.polyval(np.polyder(Dp), 0))/np.polyval(Dp, 0)**2
        res = [np.polyval(Np, pi)/np.polyval(np.polyder(Dp), pi) for pi in poles]
        ramp = lambda t: np.real(G0*t + dG0 + sum(r*np.exp(pi*t)/pi**2 for r, pi in zip(res, poles)))
        step = lambda t: np.real(G0 + sum(r*np.exp(pi*t)/pi for r, pi in zip(res, poles)))
        return dict(G0=G0, dG0=dG0, res=[float(np.real(r)) for r in res], ramp=ramp, step=step)
    return dict(poles=[float(np.real(x)) for x in poles], o=build(No), Q=build(NQ), Cs=Cs, Rg=Rg, Ri=Ri)

# ----------------------------------------------------------------------------------------
# Finite-difference (method of lines) check; also allows TIME-VARYING wind
# ----------------------------------------------------------------------------------------
def fd_solve(p_of_t, t_end, N=40):
    p0 = p_of_t(0.0)
    dz = p0["d"]/N
    kinks_o = T_OUT_KINKS
    def To(t):
        return sum(a*max(t - tk, 0) for tk, a in kinks_o)
    def rhs(t, y):
        p = p_of_t(t)
        T, Ta = y[:N], y[N]
        dk = diffuse_kinks(p)
        Q = p["Q_dogs"] + sum(a*max(t - tk, 0) for tk, a in dk)
        Ri = p["R_fin"] + 1/p["h_in"]
        dT = np.empty(N)
        lap = np.zeros(N)
        lap[1:-1] = T[2:] - 2*T[1:-1] + T[:-2]
        q_bot = (To(t) - T[0])/(1/p["h_g"] + dz/(2*p["k"]))
        q_top = (T[-1] - Ta)/(Ri + dz/(2*p["k"]))
        lap[0] = T[1] - T[0]
        lap[-1] = T[-2] - T[-1]
        dT[:] = p["k"]*lap/dz**2
        dT[0] += q_bot/dz
        dT[-1] -= q_top/dz
        dT /= p["rhoc"]
        dTa = (p["A_f"]*q_top + p["K_env"]*(To(t) - Ta) + Q)/p["C_a"]
        return np.concatenate([dT, [dTa]])
    t = np.linspace(0, t_end, 361)
    q0 = -p0["theta_a0"]/p0["R_tot"]                   # steady upward flux with AC on
    z = (np.arange(N) + 0.5)*dz
    y0 = np.concatenate([-q0*(1/p0["h_g"] + z/p0["k"]), [p0["theta_a0"]]])
    sol = solve_ivp(rhs, (0, t_end), y0, t_eval=t, method="BDF", max_step=300)
    return sol.t, sol.y[N], sol.y[N - 1]

def wind_profile(t):
    """Light & variable early, W 10 mph by afternoon (sea breeze)."""
    h = 6 + t/HR
    return np.interp(h, [6, 9, 12, 15], [1.0, 2.0, 4.0, 4.5])


# ----------------------------------------------------------------------------------------
if __name__ == "__main__":
    t = np.linspace(0, 9*HR, 361)
    clock = 6 + t/HR
    res, summary = {}, {}
    for name, v in WIND_10M.items():
        p = params(v)
        res[name] = solve_laplace(p, t)
        summary[name] = dict(K_env=p["K_env"], UA_glass=p["UA_glass"], UA_opq=p["UA_opq"],
                             UA_inf=p["UA_inf"], h_o=p["h_o"], h_g=p["h_g"], ACH=p["ACH"])
    p = params()
    r = res["forecast"]
    # route decomposition at each time (W)
    Ta, To = r["Th_a"], r["Th_o"]
    routes = {
        "1 floor slab (garage)": p["A_f"]*r["q_floor"],
        "3 glazing conduction": p["UA_glass"]*(To - Ta),
        "3 opaque walls": p["UA_opq"]*(To - Ta),
        "3 infiltration (wind)": p["UA_inf"]*(To - Ta),
        "3b skylight (north glass + wall sol-air)": r["Q_dif"],
        "dogs": p["Q_dogs"]*np.ones_like(t),
        "2 courtyard slab (flush, upper bound)": r["Q_court"],
    }
    # route 2 at separation of one unit width (25 ft)
    r_far = solve_laplace(p, t, L_court=25*FT)
    routes_far_court = r_far["Q_court"]

    # floor surface temperature (dog lying on it) and 2 ft air estimate
    T_floor = Ta + r["q_floor"]/p["h_in"]
    grad = 0.5      # K/m, assumed stable stratification (warm gains enter high, floor lags)
    T_2ft = Ta - grad*(6*FT - 2*FT)

    # lumped analytic
    lump = lumped_symbolic(p)
    Ta_lump = (shift_sum(t, lump["o"]["ramp"](t), T_OUT_KINKS, t)
               + shift_sum(t, lump["Q"]["ramp"](t), diffuse_kinks(p), t)
               + lump["Q"]["G0"]*(p["theta_a0"]/lump["Q"]["G0"])
               + (p["Q_dogs"] - p["theta_a0"]/lump["Q"]["G0"])*lump["Q"]["step"](t))
    # FD checks
    tfd, Ta_fd, _ = fd_solve(lambda tt: params(), 9*HR)
    tfd2, Ta_fd_wind, _ = fd_solve(lambda tt: params(float(wind_profile(tt))), 9*HR)

    r_noac = solve_laplace(params(T_room0_F=T0_F), t)           # no AC overnight
    r_east = solve_laplace(params(east_glass_ft2=36.0), t)      # one unlit 9'x4' east window

    def F(x):
        return T0_F + c2f(x)

    # ---------------- figures
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    BLIND = 0.3                       # blinds/curtains closed: 30 % of the diffuse gain gets in
    Ta_bl = Ta - (1 - BLIND)*r["Th_a_dif"]
    T_floor_bl = Ta_bl + (r["q_floor"] - (1 - BLIND)*r["q_floor_dif"])/p["h_in"]
    fig, ax = plt.subplots(figsize=(8, 4.6))
    ax.plot(clock, F(To), color="#c2410c", lw=2, label="Outdoor = garage air (forecast)")
    ax.plot(clock, F(Ta), color="#1d4ed8", lw=2.2, label="Room air, mixed mean")
    ax.plot(clock, F(T_2ft), color="#0f766e", lw=1.5, label="Air at 2 ft (est., 0.5 K/m gradient)")
    ax.plot(clock, F(T_floor), color="#7c3aed", lw=1.5, label="Floor surface")
    ax.plot(clock, F(Ta_bl), color="#1d4ed8", lw=1.5, ls="--", label="Room air, blinds closed")
    ax.plot(clock, F(T_floor_bl), color="#7c3aed", lw=1.2, ls="--", label="Floor surface, blinds closed")
    ax.set_xlabel("Clock time (PDT)"); ax.set_ylabel("°F")
    ax.set_xticks(range(6, 16)); ax.grid(alpha=0.25)
    ax.legend(fontsize=8, frameon=False, loc="upper left")
    fig.tight_layout(); fig.savefig("fig_temperatures.png", dpi=150)

    fig, ax = plt.subplots(figsize=(8, 4.0))
    for name, ls in [("calm", ":"), ("forecast", "-"), ("breezy", "--")]:
        ax.plot(clock, F(res[name]["Th_a"]), color="#1d4ed8", ls=ls, lw=1.6,
                label=f"Wind {name}: {WIND_10M[name]:.1f} m/s")
    ax.plot(clock, F(r_noac["Th_a"]), color="#c2410c", lw=1.4, label="No AC overnight (room and slab start at 77 °F)")
    ax.plot(clock, F(r_east["Th_a"]), color="#64748b", lw=1.4, label="Plus one 9'x4' east window (no daylight)")
    ax.set_xlabel("Clock time (PDT)"); ax.set_ylabel("Room air, °F")
    ax.set_xticks(range(6, 16)); ax.grid(alpha=0.25); ax.legend(fontsize=8, frameon=False)
    fig.tight_layout(); fig.savefig("fig_sensitivity.png", dpi=150)

    fig, ax = plt.subplots(figsize=(8, 4.6))
    cols = ["#7c3aed", "#1d4ed8", "#64748b", "#0f766e", "#ca8a04", "#be185d", "#c2410c"]
    for (k, v), c in zip(routes.items(), cols):
        ax.plot(clock, v, lw=2, color=c, label=k)
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xlabel("Clock time (PDT)"); ax.set_ylabel("Heat into the unit, W")
    ax.set_xticks(range(6, 16)); ax.grid(alpha=0.25)
    ax.legend(fontsize=8, frameon=False, loc="lower left")
    fig.tight_layout(); fig.savefig("fig_routes.png", dpi=150)

    fig, ax = plt.subplots(figsize=(8, 4.0))
    ax.plot(clock, F(Ta - r["Th_a_court"]), lw=3, color="#1d4ed8", alpha=0.35,
            label="Laplace, exact slab (Talbot)")
    ax.plot(tfd/HR + 6, F(Ta_fd), "k--", lw=1.2, label="Finite differences (40 cells, BDF)")
    ax.plot(clock, F(Ta_lump), color="#c2410c", lw=1.2, label="Lumped 2-node, partial fractions")
    ax.plot(tfd2/HR + 6, F(Ta_fd_wind), color="#0f766e", lw=1.2,
            label="FD with time-varying wind (not LTI)")
    ax.set_xlabel("Clock time (PDT)"); ax.set_ylabel("Room air, °F")
    ax.set_xticks(range(6, 16)); ax.grid(alpha=0.25); ax.legend(fontsize=8, frameon=False)
    fig.tight_layout(); fig.savefig("fig_verification.png", dpi=150)

    idx = {h: int(np.argmin(abs(clock - h))) for h in range(6, 16)}
    out = dict(
        params={k: (float(v) if np.isscalar(v) else v) for k, v in p.items()},
        wind_cases=summary,
        lumped=dict(poles=lump["poles"], tau_h=[-1/x/HR for x in lump["poles"]],
                    Cs=lump["Cs"], Rg=lump["Rg"], Ri=lump["Ri"],
                    o={k: lump["o"][k] for k in ("G0", "dG0", "res")},
                    Q={k: lump["Q"][k] for k in ("G0", "dG0", "res")}),
        q_sun_step=r["q_sun"],
        hourly={h: dict(T_out=F(To[i]), T_air=F(Ta[i]), T_air_calm=F(res["calm"]["Th_a"][i]),
                        T_air_breezy=F(res["breezy"]["Th_a"][i]), T_2ft=F(T_2ft[i]),
                        T_floor=F(T_floor[i]), T_air_fd=F(Ta_fd[i]), T_air_fd_wind=F(Ta_fd_wind[i]),
                        T_air_noac=F(r_noac["Th_a"][i]), T_air_east=F(r_east["Th_a"][i]),
                        T_air_blinds=F(Ta_bl[i]),
                        T_floor_blinds=F(T_floor_bl[i]), T_air_nocourt=F(Ta[i] - r["Th_a_court"][i]),
                        T_air_lump=F(Ta_lump[i]),
                        **{k: float(v[i]) for k, v in routes.items()},
                        court_far=float(routes_far_court[i]))
                for h, i in idx.items()},
        energy_kWh={k: float(np.trapezoid(v, t)/3.6e6) for k, v in routes.items()},
        sun_alt={h: float(solar.altitude_deg(h)) for h in range(6, 16)},
        solar_noon=solar.solar_noon_pdt(),
    )
    with open("results.json", "w") as f:
        json.dump(out, f, indent=1, default=float)
    print(json.dumps({k: out[k] for k in ("wind_cases", "lumped", "q_sun_step", "energy_kWh")},
                     indent=1, default=float))
    for h in range(6, 16):
        d = out["hourly"][h]
        print(h, " ".join(f"{k[:14]}={v:7.1f}" for k, v in d.items()))
