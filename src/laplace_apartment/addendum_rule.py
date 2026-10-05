"""
Decision rule for other days (addendum): predicted room temperature at 3 pm, AC off from 06:00,
north blinds closed, PC off, room at 70 F at 06:00, as a function of the forecast low (06:00)
and high (15:00).  Linear system -> T_room(3pm) = c0 + c_low*T_low + c_high*T_high exactly.
Outdoor profile: half-cosine rise from low at 06:00 to high at 15:00, as hourly ramps.
"""
import numpy as np
from apartment_heat import params, G_room, invert, ramp_kinks, shift_sum, diffuse_kinks, HR, F2C

p = params(); p["blind"] = 0.3; p["Q_dogs"] = 100.0
T_END = 9*HR
hrs = np.arange(6, 16)
shape = 0.5*(1 - np.cos(np.pi*(hrs - 6)/9))            # 0 -> 1
kinks = ramp_kinks(list(hrs), list(shape))              # unit swing, in K per K of (high-low)
tb = np.linspace(0, T_END, 91)
r_o = invert(lambda s: G_room(p, s)[0]/s**2, tb)
r_Q = invert(lambda s: G_room(p, s)[1]/s**2, tb)
u_Q = invert(lambda s: G_room(p, s)[1]/s, tb)
t = np.array([T_END])
beta = shift_sum(tb, r_o, kinks, t)[0]                  # per K of (high - low)
uQ9 = u_Q[-1]
decay = 1 - uQ9/p["GQ0"]                                # fraction of initial offset left
gains_K = p["Q_dogs"]*uQ9 + shift_sum(tb, r_Q, diffuse_kinks(p), t)[0]
# T_room = T_low + (70 - T_low)*decay + beta*(T_high - T_low) + gains
c_low = 1 - decay - beta
c_high = beta
c0 = 70*decay + gains_K*9/5
print(f"T_room(3pm) = {c0:.2f} + {c_low:.3f}*T_low + {c_high:.3f}*T_high   [F]")
print(f"decay={decay:.3f} beta={beta:.3f} gains={gains_K*9/5:.2f} F")
f = lambda lo, hi: c0 + c_low*lo + c_high*hi
for lo in (60, 65, 70, 75):
    print(lo, [round(f(lo, hi), 1) for hi in (75, 80, 85, 90, 95, 100)])
for lim in (78, 80, 85):
    print("limit", lim, {lo: round((lim - c0 - c_low*lo)/c_high, 1) for lo in (60, 65, 70, 75)})
print("today (77, 99):", round(f(77, 99), 1))
