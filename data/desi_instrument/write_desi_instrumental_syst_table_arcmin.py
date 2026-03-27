#!/usr/bin/env python
import numpy as np
from astropy.table import Table
from vega.utils import find_file



def main():
    path = "instrumental_systematics/desi-positioners.csv"
    file = find_file(path)
    print(f"Reading {file}")
    positioner_table = Table.read(file)

    xp = positioner_table["FOCAL_PLANE_X_DEG"]
    yp = positioner_table["FOCAL_PLANE_Y_DEG"]
    rpatrol = positioner_table["PATROL_RADIUS_DEG"]

    print("Compute randoms...")
    nr = 50000
    x = np.random.uniform(size=nr) * (np.max(xp + rpatrol))
    y = np.random.uniform(size=nr) * (np.max(yp + rpatrol))
    ok = np.repeat(False, nr)
    for xxp, yyp, rrp in zip(xp, yp, rpatrol) :
        ok |= ((x - xxp)**2 + (y - yyp)**2) < rrp**2
    x = x[ok]
    y = y[ok]

    print("Compute correlation...")
    #deg2mpc = comoving_distance * np.pi / 180.
    bins = np.linspace(0, 200, 401)
    nbins = bins.size - 1
    h0 = np.zeros(nbins)
    for xx, yy in zip(x, y):
        d_deg = np.sqrt((xx - x)**2 + (yy - y)**2) 
        d_arcmin = d_deg * 60.0
        t, _ = np.histogram(d_arcmin, bins=bins)
        h0 += t
    ok = (h0 > 0)
    rt = (bins[:-1] + (bins[1] - bins[0]) / 2)
    rt = rt[ok]
    xi = h0[ok] / rt  # number of random pairs scales as rt

    # add a value at 0, last measured bin + 1 step, and 1000 Mpc to avoid extrapolations
    xi_at_0 = (xi[0] - xi[1]) / (rt[0] - rt[1]) * (0 - rt[0]) + xi[0]  # linearly extrapolated to r=0
    rt = np.append(0, rt)
    xi = np.append(xi_at_0, xi)
    rt = np.append(rt, [rt[-1] + bins[1] - bins[0], 1000.])
    xi = np.append(xi, [0, 0])
    xi /= np.max(xi)  # norm

    t = Table()
    t["theta_arc"] = rt
    t["xi_noise"] = xi
    filename = "desi-instrument-syst-for-forest-auto-correlation_arcmin.csv"
    t.write(filename, overwrite=True)
    print("wrote ", filename)


if __name__ == '__main__':
    main()
