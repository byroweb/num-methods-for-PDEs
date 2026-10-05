"""Clear-sky sun position / irradiance for San Diego, 5 Oct 2026 (PDT)."""
import numpy as np

LAT, LON = 32.7157, -117.1611
DOY = 278                       # 5 October

def declination(n=DOY):
    return np.radians(23.44) * np.sin(2*np.pi*(284 + n)/365)

def eq_of_time_min(n=DOY):
    B = 2*np.pi*(n - 81)/364
    return 9.87*np.sin(2*B) - 7.53*np.cos(B) - 1.5*np.sin(B)

def solar_noon_pdt():
    # PDT = UTC-7 -> reference meridian 105 W
    return 12.0 + (abs(LON) - 105.0)*4/60 - eq_of_time_min()/60

def altitude_deg(clock_h):
    H = np.radians(15*(clock_h - solar_noon_pdt()))
    phi, d = np.radians(LAT), declination()
    s = np.sin(phi)*np.sin(d) + np.cos(phi)*np.cos(d)*np.cos(H)
    return np.degrees(np.arcsin(s))

def ghi_clear(clock_h):
    """Haurwitz clear-sky global horizontal irradiance, W/m^2."""
    cz = np.sin(np.radians(altitude_deg(clock_h)))
    return np.where(cz > 0, 1098*cz*np.exp(-0.057/np.maximum(cz, 1e-3)), 0.0)

if __name__ == "__main__":
    print(f"solar noon {solar_noon_pdt():.2f} h PDT, decl {np.degrees(declination()):.2f} deg")
    for h in np.arange(6, 16, 1.0):
        print(f"{h:5.1f}  alt {altitude_deg(h):6.1f}  GHI {float(ghi_clear(h)):6.0f}")
