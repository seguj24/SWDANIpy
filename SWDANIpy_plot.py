#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 13 19:18:07 2025

@author: segu
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt
# import matplotlib as mpl

## Bayesian inversion module
from bayesbay.discretization import Voronoi1D


# Utils
import os
import sys
d_def = "/Users/seguuu/Project/02_Bayesian_inversion/SWDANI_SGmod/SWDANIpy/code_utils"
sys.path.append(f"{d_def}")
# from datacov import dcov
from read_par import read_par

try:
    get_ipython
    is_ipython = True
except NameError:
    is_ipython = False
    

####################
# Read Par file
####################
par = read_par("./Par")
# dict_keys(['Nchain', 'mcmc', 'priors', 'weights', 'datasets'])

priors = par["priors"]
weights = par['weights']
datasets = par["datasets"]

####################
# Parameters
####################
zmin, zmax, dz = priors['depth']
vsmin, vsmax, dvs = priors["vs"]
w0, w1, dw = weights

if zmax > 1:
   yscale = 'linear'
else:
    yscale = 'log'
    if zmin == 0:
        zmin = 0.001

####################
# Call results
####################
with open("./OUT/results.pkl", "rb") as f:
    _results = pickle.load(f)

results = {}
for key, chains in _results.items():
    results[key] = [s for c in chains for s in c]
    
####################
# Voronoi / Vs samples
####################
DRAW_MEAN_STD = True
DRAW_MEDIAN_CI = False
Nz   = 201
Nv   = 201
zgrid = np.linspace(zmin, zmax, Nz)
vgrid = np.linspace(vsmin, vsmax, Nv)

# Get results
voros = results['voronoi.discretization']   # list of nuclei arrays
vss   = results['voronoi.vs']              # list of vs arrays

# Thickness to Depth
thks_list  = [Voronoi1D.compute_cell_extents(n) for n in voros]
zbot_list  = [np.cumsum(thk) for thk in thks_list]

# Stair-like VS Profile
Nsamples = len(vss)
vs_profiles = np.zeros((Nsamples, Nz), dtype=float)

for i, (thk, vs) in enumerate(zip(thks_list, vss)):
    zbot = np.cumsum(thk)
    ztop = np.concatenate(([0.0], zbot[:-1]))
    for v_layer, z0, z1 in zip(vs, ztop, zbot):
        mask = (zgrid >= z0) & (zgrid < z1)
        vs_profiles[i, mask] = v_layer
    vs_profiles[i, zgrid >= zbot[-1]] = vs[-1]


# 2D PPD
hvcount = np.zeros((Nz, Nv), dtype=int)

for i in range(Nsamples):
    v_profile = vs_profiles[i, :]
    idx_v = np.searchsorted(vgrid, v_profile, side="right") - 1
    idx_v = np.clip(idx_v, 0, Nv-1)
    for iz, iv in enumerate(idx_v):
        hvcount[iz, iv] += 1

# hvprob = hvcount / hvcount.max()
hvprob = hvcount / hvcount.max(axis=1, keepdims=True).clip(min=1)

fig, ax = plt.subplots(figsize=(3,3), dpi=300)

pcm = ax.pcolormesh(
    vgrid, zgrid, hvprob,
    shading="auto",
    cmap="Grays"
)

ax.set_xlabel(r"$V_S$ (km/s)")
ax.set_ylabel("Depth (km)")
ax.invert_yaxis()
ax.set_ylim(zmax, zmin)

cbar = fig.colorbar(pcm, ax=ax, label="Normalized probability")

# MEAN_STD / MEDIAN_95%CI
mean_profile  = vs_profiles.mean(axis=0)
std_profile   = vs_profiles.std(axis=0)

median_profile = np.median(vs_profiles, axis=0)
q025 = np.quantile(vs_profiles, 0.025, axis=0)
q975 = np.quantile(vs_profiles, 0.975, axis=0)

if DRAW_MEAN_STD:
    ax.plot(mean_profile, zgrid, color='blue', label="μ", lw=0.8, alpha=0.5)
    ax.fill_betweenx(zgrid, mean_profile-std_profile, mean_profile+std_profile,
                     color='blue', alpha=0.2, label="±1 σ", lw=0.8)

if DRAW_MEDIAN_CI:
    ax.plot(median_profile, zgrid, lw=0.8, color='red', label="Median")
    ax.fill_betweenx(zgrid, q025, q975,
                     color='red', lw=0.8, alpha=0.25,
                     label="95% CI")

ax.grid(ls='--', lw=0.5, c='gray')
ax.legend(fontsize=7)

plt.tight_layout()

os.makedirs('./figure', exist_ok=True)
plt.savefig("./figure/PPD_VSprofile.png", dpi=300)

if is_ipython:
    fig.show()
else:
    plt.close(fig)  
    
####################
# Interface PDF
####################    
nlays = np.array(results["voronoi.n_dimensions"])

vals, counts = np.unique(nlays, return_counts=True)
# Normalization
pdf = counts / counts.sum()

plt.figure(figsize=(3,2))
plt.plot(vals, pdf, linestyle='-', c='k', linewidth=1)
plt.fill_between(vals, pdf, color='gray', alpha=0.3)

plt.xlabel("# Layer")
plt.ylabel("Probability Density")

plt.xlim(nlays.min(), nlays.max())
plt.ylim(0, pdf.max())

plt.grid(ls='--', c='gray', lw=0.5)
plt.savefig("./figure/PDF_interface.png", dpi=300)

if is_ipython:
    fig.show()
else:
    plt.close(fig)  

####################
# Predicted data PPD
####################
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from scipy.stats import gaussian_kde

type_to_ds = defaultdict(list)
for ds in datasets:
    full_name = ds["type"]
    dtype = full_name[:3]
    type_to_ds[dtype].append(ds)

types = sorted(type_to_ds.keys())
n_types = len(types)

fig, axes = plt.subplots(n_types, 1, figsize=(5, 3*n_types), dpi=300)
if n_types == 1:
    axes = [axes]

cbar_ax = fig.add_axes([0.25, 0.06, 0.5, 0.02])  
# [left, bottom, width, height]
global_pcm = None

for ax, dtype in zip(axes, types):
    for ds in type_to_ds[dtype]:
        name = ds["type"]
        t    = ds["t"]
        obs  = ds["obs"]
        std  = np.sqrt(ds["var"])

        key = f"{name}.dpred"
        if key not in results:
            continue

        dpred = np.array(results[key])
        Nsamples, Nt = dpred.shape

# 2D PPD
        y_min = dpred.min()
        y_max = dpred.max()
        margin = 0.05 * (y_max - y_min + 1e-8)
        y_min -= margin
        y_max += margin

        Ny = 200
        ygrid = np.linspace(y_min, y_max, Ny)
        pdf_map = np.zeros((Ny, Nt))

        for j in range(Nt):
            col = dpred[:, j]
            if np.allclose(col, col[0]):
                idx = np.searchsorted(ygrid, col[0])
                idx = np.clip(idx, 0, Ny-1)
                pdf_map[idx, j] = 1.0
            else:
                kde = gaussian_kde(col)
                pdf_map[:, j] = kde(ygrid)

        pdf_map_max = pdf_map.max(axis=0, keepdims=True)
        pdf_map_max[pdf_map_max == 0] = 1.0
        pdf_map_norm = pdf_map / pdf_map_max

        # density heatmap
        pcm = ax.pcolormesh(t, ygrid, pdf_map_norm, shading="auto",
            cmap="Greys", alpha=0.85)
        global_pcm = pcm

# Observed data
        ax.errorbar(t, obs, yerr=std, fmt="o", markersize=3, capsize=2,
                    label=f"{name} μ ± σ")

    ax.set_ylabel("Velocity (km/s)")
    if dtype in ["RPV", "RGV", "LPV", "LGV"]:
        ax.set_xlabel("Period (s)")
    elif dtype in ["PRF"]:
        ax.set_xlabel("Time (s)")
    ax.grid(ls='--', c='gray', lw=0.5)

    handles, labels = ax.get_legend_handles_labels()
    uniq = dict(zip(labels, handles))
    ax.legend(uniq.values(), uniq.keys(), fontsize=8, loc="best")

# Color bar
cbar = fig.colorbar(global_pcm, cax=cbar_ax, orientation="horizontal")
cbar.set_label("Normalized density")

# plt.tight_layout(rect=[0, 0.07, 1, 1])  # 하단 컬러바 공간 확보
plt.savefig("./figure/PPD_predicted.png", dpi=300)

if is_ipython:
    fig.show()
else:
    plt.close(fig)  

####################
# Data Weight
####################
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

w_keys = [k for k in results.keys() if k.endswith(".w")]
w_keys = sorted(w_keys)

n_w = len(w_keys)
fig, axes = plt.subplots(n_w, 1, figsize=(4, 2.5*n_w), 
                         sharex=True, dpi=300)
if n_w == 1:
    axes = [axes]

for ax, key in zip(axes, w_keys):
    w_samp = np.array(results[key]).reshape(-1)

# KDE
    kde = gaussian_kde(w_samp)
    x_min, x_max = w_samp.min(), w_samp.max()
    margin = 0.1 * (x_max - x_min + 1e-8)
    x = np.linspace(x_min - margin, x_max + margin, 400)
    pdf = kde(x)
# Normalization 
    # pdf /= np.trapz(pdf, x)
    # pdf /= pdf.max()
    
    ax.plot(x, pdf, lw=1.5)
    ax.fill_between(x, 0, pdf, alpha=0.3)

    ax.set_xlim(w0, w1)
    ax.set_ylim(0, pdf.max())
    
    ax.text(0.98, 0.95, key[:-2],
        transform=ax.transAxes,
        ha='right', va='top',
        fontsize=10)
    
    ax.grid(ls='--', lw=0.5, c='gray')
axes[-1].set_xlabel("w")

plt.tight_layout()
plt.savefig("./figure/PDF_data_weight.png", dpi=300)


if is_ipython:
    fig.show()
else:
    plt.close(fig)  
