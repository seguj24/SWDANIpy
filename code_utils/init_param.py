#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 13 16:32:53 2025

@author: segu
"""

import numpy as np

def init_voro(priors, Zvoro1): #, rand_seed=176257
    # priors =
    # {'depth': (0.0, 1.0, 0.01),
    #  'vs': (0.0, 5.0, 0.05),
    #  'xi': (0.9, 1.1, 0.01, 1.0, 0.01),
    #  'vpvs': (1.7, 1.9, 0.01, 0.0, 0.05),
    #  'rho': (1.0, 4.0, 0.01, 0.0, 0.05),
    #  'nvoro': (3, 15),
    #  'rand': (12345.0, 500.0)}
    
    nvmin, nvmax = map(int, priors['nvoro'])
    vsmin, vsmax, _ = map(float, priors['vs'])
    zmin, zmax, _ = map(float, priors['depth'])
    
    # rng = np.random.default_rng(int(rand_seed))
    
    # nlay = rng.integers(nvmin, nvmax + 1)
    nlay = np.random.randint(nvmin, nvmax+1)
    
    # 2) nuclei (depth) 생성
    # nuclei = rng.uniform(zmin, zmax, size=nlay)
    nuclei = np.random.uniform(zmin, zmax, size=nlay)
    nuclei[0] = 0.01   # Fortran: param%depth%val(1) = 0.010
    
    nuclei = np.clip(nuclei, zmin, zmax)
    nuclei.sort()
    
    vs_range = abs(vsmax - vsmin)
    z_range  = abs(zmax - zmin)
    
    vs_vals = (
        0.6 * vs_range 
        / z_range * nuclei
        + vsmin
        + 0.05 * vs_range
        )

    return nuclei, vs_vals 

# priors = {'depth': (0.0, 1.0, 0.01),
#  'vs': (0.0, 5.0, 0.05),
#  'xi': (0.9, 1.1, 0.01, 1.0, 0.01),
#  'vpvs': (1.7, 1.9, 0.01, 0.0, 0.05),
#  'rho': (1.0, 4.0, 0.01, 0.0, 0.05),
#  'nvoro': (3, 15),
#  'rand': (12345.0, 500.0)}


# nuclei, vs_vals = init_voro(priors)



# Fortran SWDANI
# param%vs%val(i) = 0.6 * abs(param%vs%max - param%vs%min) /                          &
#                   abs(param%depth%max - param%depth%min) * param%depth%val(i) +     &
#                   param%vs%min + 0.05 * abs(param%vs%min - param%vs%max)
