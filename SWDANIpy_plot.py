#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 13 19:18:07 2025

@author: segu
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

## Bayesian inversion module
from bayesbay.discretization import Voronoi1D

# Utils
import sys
d_def = "/Users/seguuu/Project/02_Bayesian_inversion/SWDANI_SGmod/SWDANIpy/code_utils"
sys.path.append(f"{d_def}")
# from datacov import dcov
from read_par import read_par

print("########## ONLY PLOT VsProfile yet")  

####################
# Read Par file
####################
par = read_par("./Par")
# dict_keys(['Nchain', 'mcmc', 'priors', 'weights', 'datasets'])

priors = par["priors"]


datasets = par["datasets"]

yscale = 'log'


####################
# parameterization
####################
zmin, zmax, dz = priors['depth']
vsmin, vsmax, dvs = priors["vs"]

# -------------------------
# 결과 불러오기
# -------------------------
with open("./OUT/results.pkl", "rb") as f:
    _results = pickle.load(f)

results = {}
for key, chains in _results.items():
    results[key] = [s for c in chains for s in c]

# -------------------------
# Voronoi / Vs samples
# -------------------------
voros = results['voronoi.discretization']   # list of nuclei arrays
thks = [Voronoi1D.compute_cell_extents(n) for n in voros]
vss = results['voronoi.vs']                           # list of vs arrays


# 깊이 그리드
# dz = 0.001                    # depth bin 간격 (zmax와 같은 단위)
nz = 200
# z_edges = np.arange(0.0, zmax + dz, dz)   # bin 경계
# z_centers = 0.5 * (z_edges[:-1] + z_edges[1:])  # center (층이 지나가는지 판정용)
z_edges = np.logspace(np.log10(1e-3), np.log10(zmax), nz)  # 예: 800 bins 로그 등분
z_centers = 0.5 * (z_edges[:-1] + z_edges[1:])

# Vs 그리드
n_vs = 100
v_edges = np.linspace(vsmin, vsmax, n_vs)

# 2D 카운트 (depth × Vs)
density = np.zeros((len(z_edges) - 1, len(v_edges) - 1), dtype=float)

# 인터페이스 깊이 카운트 (1D)
interface_density = np.zeros(len(z_edges) - 1, dtype=float)
# breakpoint()
for thk, vs in zip(thks, vss):
    thk = np.asarray(thk)
    vs  = np.asarray(vs)

    # 두께 합 → zmax 로 스케일
    total_thk = np.sum(thk)
    # if total_thk == 0:
    #     continue

    # 깊이 인터페이스 (0 ~ zmax 근처까지)
    z_interfaces = np.concatenate(([0.0], np.cumsum(thk)))
    z_interfaces[-1] = zmax

    # --- 2D density 카운트 ---
    for j in range(len(vs)):
        z_top = z_interfaces[j]
        z_bot = z_interfaces[j+1]
        v_layer = vs[j]

        depth_mask = (z_centers >= z_top) & (z_centers < z_bot)
        if not np.any(depth_mask):
            continue

        iv = np.searchsorted(v_edges, v_layer) - 1
        if iv < 0 or iv >= len(v_edges) - 1:
            continue

        iz_indices = np.where(depth_mask)[0]
        density[iz_indices, iv] += 1.0

    # --- interface 깊이 카운트 ---
    # 맨 위(0)와 맨 아래(zmax 근처)는 제외하고 내부 인터페이스만 사용
    for z_int in z_interfaces[1:-1]:
        iz = np.searchsorted(z_edges, z_int) - 1
        if 0 <= iz < len(interface_density):
            interface_density[iz] += 1.0


####################
# Median model
####################
density_counts = density.copy()
v_centers = 0.5 * (v_edges[:-1] + v_edges[1:])

n_depth = density_counts.shape[0]
median_vs = np.full(n_depth, np.nan)
vs_low   = np.full(n_depth, np.nan)   # 2%
vs_high  = np.full(n_depth, np.nan)   # 98%

for iz in range(n_depth):
    row = density_counts[iz, :]
    s = row.sum()
    if s <= 0:
        continue

    pdf = row / s
    cdf = np.cumsum(pdf)

    # 2%, 50%, 98% 위치
    for q, arr in zip([0.025, 0.5, 0.975], [vs_low, median_vs, vs_high]):
        idx = np.searchsorted(cdf, q)
        if idx >= len(v_centers):
            idx = len(v_centers) - 1
        arr[iz] = v_centers[idx]



norm_ppd = True
if norm_ppd:
    max_vals = density.max(axis=1, keepdims=True)
    density = density / np.where(max_vals == 0, 1, max_vals)


interface_density_norm = interface_density / np.max(interface_density) if interface_density.max() > 0 else interface_density

fig, (ax_intf, ax_vs) = plt.subplots(
    ncols=2, 
    sharey=True,
    figsize=(4, 3), 
    dpi = 300,
    gridspec_kw={'width_ratios': [1, 3]}
    )

# CMAP
facecolor = "white"
colors = [facecolor, "white", "yellow", "orange", "tomato", "red"]  
color_positions = [0.0, 0.0001, 0.25, 0.5, 0.8, 1.0]
cmap = mpl.colors.LinearSegmentedColormap.from_list("", list(zip(color_positions, colors)))

####################
# density
####################
img = ax_vs.pcolormesh(
    v_edges,
    z_edges,
    density,
    shading="auto",
    cmap="binary"
    )

ax_vs.set_xlabel("Vs")

cbar = fig.colorbar(img, ax=ax_vs)
cbar.set_label("Normalized density")

####################
# Median
####################
ax_vs.plot(median_vs, z_centers, 'r-', lw=1, label='median')
ax_vs.plot(vs_low, z_centers, 'r--', lw=1, label='95% CI')
ax_vs.plot(vs_high, z_centers,'r--', lw=1)
ax_vs.legend(loc='upper right')

####################
# Interface probability
####################
ax_intf.plot(interface_density_norm, z_centers, 'k-', lw=1.5)

ax_intf.set_xlim(0, 1.05)
ax_intf.set_xlabel("Interface probability")
ax_intf.set_ylabel("Depth [m]")

for ax in [ax_intf, ax_vs]:
    ax.invert_yaxis()
    ax.set_yscale(yscale)
    if yscale == 'log' and zmin == 0:
        zmin = 0.001
    ax.set_ylim(zmax, zmin)
    
    ax.grid(ls='--', lw=0.5, c='gray')
    
plt.tight_layout()
plt.show()




