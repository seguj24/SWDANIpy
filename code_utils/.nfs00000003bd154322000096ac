#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 12 19:48:30 2025

@author: segu
"""


import numpy as np
from pathlib import Path
from functools import partial

import sys
d_def = '/home/seguuu/Project/02_Bayesian_inversion/BayesBay_TEST/code_utils'
sys.path.append(f"{d_def}")
from forward import forward_PVswd, forward_GVswd, forward_ell
from datacov import init_dcov


def read_par(par_path="./Par", static_property=False):
    par_path = Path(par_path)
    lines = par_path.read_text(encoding="utf-8").splitlines()
    
    def nxt_tokens(it):
        # 빈줄/전행주석 스킵 + 인라인 주석 제거
        for s in it:
            s = s.strip()
            if not s or s.startswith(("!", "#")):
                continue
            s = s.split("!")[0].split("#")[0].strip()
            if s:
                return s.split()
        raise EOFError("Par EOF")
    
    it = iter(lines)
    
    t = nxt_tokens(it)
    burnin, sample, skip = map(float, t[:3])
    Ncore = int(nxt_tokens(it)[0])
    sampler = str(nxt_tokens(it)[0])
    if sampler not in ['PT', 'SA', 'None']:
        print(f">>> Warning: Invalid sampler '{sampler}' detected.")
        print(">>> Valid options are: PT, SA, or None.")
        print(">>> Setting sampler to 'None' by default.\n")
        sampler = "None"
    
    depth_min, depth_max, depth_delta = map(float, nxt_tokens(it)[:3])
    vs_min, vs_max, vs_delta         = map(float, nxt_tokens(it)[:3])
    xi_min, xi_max, xi_delta, xi_sw  = map(float, nxt_tokens(it)[:4])
    vpvs_min, vpvs_max, vpvs_delta, vpvs_sw = map(float, nxt_tokens(it)[:4])
    rho_min, rho_max, rho_delta, rho_sw     = map(float, nxt_tokens(it)[:4])
    vs_transd, xi_transd, vpvs_transd, rho_transd = map(float, nxt_tokens(it)[:4])
    nvoro_min, nvoro_max = map(int, nxt_tokens(it)[:2])
    rand_seed, rand_rmul = map(float, nxt_tokens(it)[:2])
    
    
    tk = nxt_tokens(it)
    w0, w1, dw = map(float, tk[:3])
    
    # 데이터 개수: 첫 토큰만 사용
    ndata = int(float(nxt_tokens(it)[0]))
        
    datasets = []
    
    SWD = {"RPV","RGV","LPV","LGV","ELL"}
    type_count = {k: 0 for k in SWD}
    for _ in range(ndata):
        cindex = nxt_tokens(it)[0].upper()
        if cindex not in SWD:
            raise ValueError(f"Invalid data type: {cindex}")
        type_count[cindex] += 1
        type_label = f"{cindex}{type_count[cindex]:02d}"
    
        filename = " ".join(nxt_tokens(it)).strip()
        filename = filename.strip("'\"") 
        fpath = par_path.parent / filename
    
        arr = np.loadtxt(fpath, comments=("#","!"))
        if arr.ndim == 1:
            arr = arr[None, :]
            
        # Sort data by period (ascending order)
        idx = np.argsort(arr[:,0])
        arr = arr[idx, :]
        t, obs, std = arr[:,0], arr[:,1], arr[:,2]
        
        if cindex == 'RPV':
            forward_fn = partial(forward_PVswd, dperi=t, static_property=static_property, wave='rayleigh', mode=0)
        elif cindex == 'LPV':
            forward_fn = partial(forward_PVswd, dperi=t, static_property=static_property, wave='love', mode=0)
        elif cindex == 'RGV':
            forward_fn = partial(forward_GVswd, dperi=t, static_property=static_property, wave='rayleigh', mode=0)
        elif cindex == 'LGV':
            forward_fn = partial(forward_GVswd, dperi=t, static_property=static_property, wave='love', mode=0)
        elif cindex == 'ELL':
            forward_fn = partial(forward_ell, dperi=t, static_property=static_property, wave='rayleigh', mode=0)
        # Ref. "./forward.py"
            
        dcov, dcov_inv = init_dcov(arr)
        # Ref. "./datacov.py"

        datasets.append({
            "type": type_label,
            "file": str(fpath),
            "t": t,
            "obs": obs,
            "var": std**2,
            "dcov": dcov,
            "dcov_inv": dcov_inv,
            "forward": forward_fn
            })
    
    par =  {
        "Nchain": Ncore,
        "mcmc": {"burnin": burnin, "sample": sample, "skip": skip},
        "sampler": sampler,
        "transD": {"vs": vs_transd, "xi": xi_transd, "vpvs": vpvs_transd, "rho":rho_transd},
        "priors": {
            "depth": (depth_min, depth_max, depth_delta),
            "vs": (vs_min, vs_max, vs_delta),
            "xi": (xi_min, xi_max, xi_delta, xi_sw, xi_transd),
            "vpvs": (vpvs_min, vpvs_max, vpvs_delta, vpvs_sw, vpvs_transd),
            "rho": (rho_min, rho_max, rho_delta, rho_sw, rho_transd),
            "nvoro": (nvoro_min, nvoro_max),
            "rand": (rand_seed, rand_rmul),
            },
        "weights": (w0, w1, dw),
        "datasets": datasets,
        }

    return par





# def read_par(par_path="./Par"):
#     par_path = Path(par_path)
#     lines = par_path.read_text(encoding="utf-8").splitlines()

#     def nxt_tokens(it):
#         # 빈줄/전행주석 스킵 + 인라인 주석 제거
#         for s in it:
#             s = s.strip()
#             if not s or s.startswith(("!", "#")):
#                 continue
#             s = s.split("!")[0].split("#")[0].strip()
#             if s:
#                 return s.split()
#         raise EOFError("Par EOF")

#     it = iter(lines)

#     # MCMC
#     t = nxt_tokens(it); updcov, iudc, burnin, sample, skip = map(float, t[:5])

#     # Priors 헤더 1줄
#     _ = nxt_tokens(it)

#     depth_min, depth_max, depth_delta = map(float, nxt_tokens(it)[:3])
#     vs_min, vs_max, vs_delta         = map(float, nxt_tokens(it)[:3])
#     xi_min, xi_max, xi_delta, xi_sw  = map(float, nxt_tokens(it)[:4])
#     vpvs_min, vpvs_max, vpvs_delta, vpvs_sw = map(float, nxt_tokens(it)[:4])
#     rho_min, rho_max, rho_delta, rho_sw     = map(float, nxt_tokens(it)[:4])
#     vs_transd, xi_transd, vpvs_transd, rho_transd = map(float, nxt_tokens(it)[:4])
#     nvoro_min, nvoro_max = map(int, nxt_tokens(it)[:2])
#     rand_seed, rand_rmul = map(float, nxt_tokens(it)[:2])

#     # 가중치: 앞의 3개만 사용
#     tk = nxt_tokens(it)
#     w0, w1, wd = map(float, tk[:3])

#     # 데이터 개수: 첫 토큰만 사용
#     ndata = int(float(nxt_tokens(it)[0]))

#     datasets = []
#     valid = {"RPV","RGV","LPV","LGV"}
#     for _ in range(ndata):
#         cindex = nxt_tokens(it)[0].upper()
#         if cindex not in valid:
#             raise ValueError(f"Invalid data type: {cindex}")
#         filename = " ".join(nxt_tokens(it))
#         fpath = par_path.parent / filename

#         arr = np.loadtxt(fpath, comments=("#","!"))
#         if arr.ndim == 1:
#             arr = arr[None, :]
#         t, obs, std = arr[:,0], arr[:,1], arr[:,2]
#         datasets.append({"type": cindex, "file": str(fpath), "t": t, "obs": obs, "std": std})

#     return {
#         "mcmc": {"updcov": updcov, "iudc": iudc, "burnin": burnin, "sample": sample, "skip": skip},
#         "priors": {
#             "depth": (depth_min, depth_max, depth_delta),
#             "vs": (vs_min, vs_max, vs_delta),
#             "xi": (xi_min, xi_max, xi_delta, xi_sw, xi_transd),
#             "vpvs": (vpvs_min, vpvs_max, vpvs_delta, vpvs_sw, vpvs_transd),
#             "rho": (rho_min, rho_max, rho_delta, rho_sw, rho_transd),
#             "nvoro": (nvoro_min, nvoro_max),
#             "rand": (int(rand_seed), rand_rmul),
#         },
#         "weights": (w0, w1, wd),
#         "datasets": datasets,
#     }