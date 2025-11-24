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

import pyhk  

import sys
d_def = '/home/seguuu/Project/02_Bayesian_inversion/BayesBay_TEST/code_utils'
sys.path.append(f"{d_def}")
from conv_property import vs2vp, vp2rho
import numpy as np



def get_model_property(state, prior_switches):
    voronoi = state["voronoi"]
    voronoi_sites = voronoi["discretization"]
    thk = Voronoi1D.compute_cell_extents(voronoi_sites)
    vs = voronoi["vs"]
    
    vpvs_sw = prior_switches["vpvs"]
    rho_sw  = prior_switches["rho"]
    # xi_sw   = prior_switches["xi"]
    
    if vpvs_sw == 0:                # perturb
        vpvs = voronoi["vpvs"]
        vp   = vpvs*vs
    elif vpvs_sw == 1:              # fixed
        vpvs = np.full_like(vs, 1.75)
        vp   = vpvs*vs
    else:                           # empirical (Brocher, 2006)
        vp   = vs2vp(vs)

    if rho_sw == 0:                # perturb
        rho = voronoi["rho"]
    elif rho_sw == 1:              # fixed
        rho = np.full_like(vs, 2.0)
    else:                           # empirical (Brocher, 2006)
        rho = vp2rho(vp)
        # Ref. "./conv_property.py"

    return thk, vs, vp, rho


def forward_PVswd(state, dperi, prior_switches, wave='rayleigh', mode=0):
    
    thk, vs, vp, rho = get_model_property(state, prior_switches)
    
    pd = PhaseDispersion(thk, vp, vs, rho)
    pv_pred = pd(dperi, mode=mode, wave=wave).velocity
    
    if len(dperi) != len(pv_pred):
        pv_pred = np.full(len(dperi), -1e+99, dtype=float)    
    
    return pv_pred


def forward_GVswd(state, dperi, prior_switches, wave='rayleigh', mode=0):
    """
    dperi: 주기[s] 배열
    wave : 'rayleigh' 또는 'love'
    """

    thk, vs, vp, rho = get_model_property(state, prior_switches)
        
    gd = GroupDispersion(thk, vp, vs, rho)
    gv_pred = gd(dperi, mode=mode, wave=wave).velocity
    
    if len(dperi) != len(gv_pred):
        # If the proposed model cannot generate predictions across the full
        # requested period range, fill with a very small value so that the
        # log-likelihood becomes extremely small (penalizing the model).
        gv_pred = np.full(len(dperi), -1e+99, dtype=float)    
        
    return gv_pred


def forward_ell(state, dperi, prior_switches, wave='rayleigh', mode=0):

    thk, vs, vp, rho = get_model_property(state, prior_switches)
        
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
    el_pred = np.abs(ell(dperi, mode=mode).ellipticity)
    if len(dperi) != len(el_pred):
        el_pred = np.full(len(dperi), -1e+99, dtype=float)    
    # print("LEN dperi:", len(dperi), "LEN d_pred:", len(el_pred))

    return el_pred

def forward_rf(state, dtime, slowness, gauss, prior_switches):

    thk, vs, vp, rho = get_model_property(state, prior_switches)
    vpvs = vp/vs
    
    tintv   = dtime[1] - dtime[0]     # time interval
    tsft    = -dtime[0]
    tdur   = dtime[-1]-dtime[0]
    
    rf_pred = pyhk.rfcalc(ps=0, thik=thk, beta=vs, kapa=vpvs, p=slowness, duration=tdur, dt=tintv, shft=tsft, gauss=gauss)
    
    if len(dtime) != len(rf_pred):
        rf_pred = np.full(len(dtime), -1e+99, dtype=float)    
    
    return rf_pred


