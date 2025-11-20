#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 13 10:20:03 2025

@author: segu
"""

import numpy as np

from bayesbay import Target


def _get_target_forward(datasets, weight):

    w0, w1, dw = weight
    targets = []
    forwards = []
    
    for ds in datasets:
        name   = ds["type"]      # 예: 'RPV01', 'RGV01', ...
        obs0   = ds["obs"]
        # var0   = ds["var"]
        forward = ds["forward"]  # forward 계산 식; read_pard 참고
        # C   = ds["dcov"] 
        Cinv = ds["dcov_inv"]
        
        target = Target(
            name = name,
            dobs = obs0,
            covariance_mat_inv=Cinv,
            noise_is_correlated=True,
            # std_min=w0,
            # std_max=w1,
            std_perturb_std=dw,
            )
        
        targets.append(target)
        forwards.append(forward)
        
    return targets, forwards


# logL_trace = []   # 전역 리스트
def Hi_loglike(state, datasets):
    
    # BAD = 1e9
    logL_total = 0.0

    for ds in datasets:
        name   = ds["type"]      # 예: 'RPV01', 'RGV01', ...
        obs0   = ds["obs"]
        # var0   = ds["var"]
        forward = ds["forward"]  # forward 계산 식; read_pard 참고
        C   = ds["dcov"] 
        Cinv = ds["dcov_inv"]

        Ndat = len(obs0)

        try:
            w = float(state[name]["w"][0])
        except (KeyError, IndexError, TypeError):
            w = 0.0
        
        dpred = forward(state)
        state.cache[f"{name}.dpred"] = dpred
        
        
        if len(obs0) != len(dpred):
            logL=-1e30
            return logL
        
        resi = obs0 - dpred
        
        scale_factor = np.exp(2.0 * w)
        
        # C_scaled^{-1} = (1/s) * C^{-1}
        # quad = r^T * C^-1 * r
        quad = (resi @ (Cinv @ resi)) / scale_factor

        # log |C_scaled| = N * log(s) + log |C|
        sign, logdet_C = np.linalg.slogdet(C)
        
        logdet_scaled = Ndat * np.log(scale_factor) + logdet_C

        logL = -0.5 * (
            Ndat * np.log(2.0 * np.pi)
            + logdet_scaled
            + quad
            )


        
        if not np.isfinite(logL):
            return -1e30
        
        logL_total += logL
    
    state.cache["logLikelihood"] = logL_total
    # logL_trace.append(logL_total)
    return logL_total

