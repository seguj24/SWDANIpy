#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 12 20:59:23 2025

@author: segu
"""

## Bayesian inversion module
from bayesbay.discretization import Voronoi1D

## forward module
from disba import PhaseDispersion, GroupDispersion, Ellipticity

import sys
d_def = '/home/seguuu/Project/02_Bayesian_inversion/BayesBay_TEST/code_utils'
sys.path.append(f"{d_def}")

from conv_property import vs2vp, vp2rho
import numpy as np


def forward_PVswd(state, dperi, static_property=True, wave='rayleigh', mode=0):
    voronoi = state["voronoi"]
    voronoi_sites = voronoi["discretization"]
    thk = Voronoi1D.compute_cell_extents(voronoi_sites)
    vs = voronoi["vs"]
    if static_property:
        vpvs= np.array([1.75]*len(vs))
        vp = vpvs/vs
        rho = np.array([2.0]*len(vs))
    else:
        vp = vs2vp(vs)
        rho = vp2rho(vp)
    # Ref. "./conv_property.py"

    pd = PhaseDispersion(thk, vp, vs, rho)
    pv_pred = pd(dperi, mode=mode, wave=wave).velocity
    
    return pv_pred


def forward_GVswd(state, dperi, static_property=True, wave='rayleigh', mode=0):
    """
    dperi: 주기[s] 배열
    wave : 'rayleigh' 또는 'love'
    """
    voronoi = state["voronoi"]
    voronoi_sites = voronoi["discretization"]
    thk = Voronoi1D.compute_cell_extents(voronoi_sites)
    vs = voronoi["vs"]
    if static_property:
        vpvs= np.array([1.75]*len(vs))
        vp = vpvs/vs
        rho = np.array([2.0]*len(vs))
    else:
        vp = vs2vp(vs)
        rho = vp2rho(vp)

    gd = GroupDispersion(thk, vp, vs, rho)
    gv_pred = gd(dperi, mode=mode, wave=wave).velocity  # [same units as disba 반환]
    return gv_pred


def forward_ell(state, dperi, static_property=True, wave='rayleigh', mode=0):
    voronoi = state["voronoi"]
    voronoi_sites = voronoi["discretization"]
    thk = Voronoi1D.compute_cell_extents(voronoi_sites)
    vs = voronoi["vs"]
    if static_property:
        vpvs= np.array([1.75]*len(vs))
        vp = vpvs/vs
        rho = np.array([2.0]*len(vs))
    else:
        vp = vs2vp(vs)
        rho = vp2rho(vp)

    
    ell = Ellipticity(thk, vp, vs, rho, algorithm='dunkin', dc=0.005)
    """
    thickness (array_like) – Layer thickness (in km).
    velocity_p (array_like) – Layer P-wave velocity (in km/s).
    velocity_s (array_like) – Layer S-wave velocity (in km/s).
    density (array_like) – Layer density (in g/cm3).
    algorithm (str {'dunkin', 'fast-delta'}, optional, default 'dunkin') –
    Algorithm to use for computation of Rayleigh-wave dispersion:
    ’dunkin’: Dunkin’s matrix (adapted from surf96),
    ’fast-delta’: fast delta matrix (after Buchen and Ben-Hador, 1996).
    dc (scalar, optional, default 0.005) – Phase velocity increment for root finding.
    """
    el_pred = ell(dperi, mode=mode).ellipticity
    print("LEN dperi:", len(dperi), "LEN d_pred:", len(el_pred))

    return np.abs(el_pred)



# def forward_swd(state, dperi, LorP, GorP, wave='rayleigh', mode=0):
#     voronoi = state["voronoi"]
#     voronoi_sites = voronoi["discretization"]
#     thk = Voronoi1D.compute_cell_extents(voronoi_sites)
#     vs = voronoi["vs"]
#     vp = vs2vp(vs)
#     rho = vp2rho(vp)    
#     if GorP == "group":
#         if LorP == "love":
            
