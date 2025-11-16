#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 14 11:09:05 2025

@author: segu
"""

from bayesbay.prior import CustomPrior
import numpy as np

def custom_prior_vs(zmin, zmax, vsmin,vsmax, vs_transd):
    
    # transd = par["transD"]
    
    # priors = par["priors"]
    
    # zmin, zmax, dz = priors['depth']
    # vsmin, vsmax, dvs = priors["vs"]
    # vs_transd = transd["vs"]         # ! 이 VS_TRANSD가 적절한건지 확인할 필요 있음.
    
    depth_deep = zmax
    vs_min_shallow = vsmin
    vs_max_shallow = vsmin + ((vsmax - vsmin) * 0.8)
    vs_min_deep = vsmax - ((vsmax - vsmin) * 0.8)
    vs_max_deep = vsmax
    
    vs_prior = CustomPrior(
        name="vs",
        log_prior=lambda value, position: (
            -np.log(
                (vs_max_shallow + (vs_max_deep - vs_max_shallow) * position / depth_deep) -
                (vs_min_shallow + (vs_min_deep - vs_min_shallow) * position / depth_deep)
                )
            if (vs_min_shallow + (vs_min_deep - vs_min_shallow) * position / depth_deep) <= value <=
               (vs_max_shallow + (vs_max_deep - vs_max_shallow) * position / depth_deep)
            else -np.inf
            ),
        sample=lambda position: np.random.uniform(
            vs_min_shallow + (vs_min_deep - vs_min_shallow) * position / depth_deep,
            vs_max_shallow + (vs_max_deep - vs_max_shallow) * position / depth_deep
            ),
        perturb_std=vs_transd
        )
    
    return vs_prior
