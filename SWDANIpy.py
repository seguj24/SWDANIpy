#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 12 18:54:15 2025

@author: segu
"""

## Bayesian inversion module
# import bayesbay as bb
from bayesbay import BayesianInversion
from bayesbay.samplers import ParallelTempering, SimulatedAnnealing
from bayesbay.likelihood import LogLikelihood
from bayesbay.discretization import Voronoi1D
from bayesbay.prior import UniformPrior
from bayesbay.parameterization import Parameterization, ParameterSpace

# Utils
from functools import partial
import sys
d_def = '/Users/seguuu/Project/02_Bayesian_inversion/SWDANIpy/code_utils'
sys.path.append(f"{d_def}")
# from datacov import dcov
from read_par import read_par
from likelihood import Hi_loglike, _get_target_forward
# from init_param import init_voro
from custom_prior import custom_prior_vs

import time, os
starttime = time.time()
print(f">>> Start time : {time.ctime(starttime)}")

####################
# Read Par file
####################
par = read_par("./Par", static_property=True)
# Ref. "./code_utils/read_par.py"
# dict_keys(['Nchain', 'mcmc', 'Sampler', 'priors', 'weights', 'datasets'])

Ncore = int(par["Nchain"])
burnin, sample, skip = int(par["mcmc"]["burnin"]), int(par["mcmc"]["sample"]), int(par["mcmc"]["skip"])
transd = par["transD"]

priors = par["priors"]
beta = par["weights"]
datasets = par["datasets"]

Ndat = int(len(datasets))

sampler_type = par["sampler"]
# PT: Parallel Tempreing
# SA: Simulated Annealing
# None: None

if Ncore == 1 and sampler_type == "PT":
    print(f">>> Chains={Ncore}, Sampler={sampler_type}")
    print(">>> Parallel Tempering needs >1 chain.")
    print(">>> Switching to Simulated Annealing.")  

custom_logL = False    
# Ref. "./code_utils/likelihood.py"
custum_prior = False
# Ref. "./code_utils/custom_prior.py"

####################
# parameterization
####################

Niter = burnin + sample

zmin, zmax, dz = priors['depth']
vsmin, vsmax, dvs = priors["vs"]
vs_transd = transd["vs"]         # ! 이 VS_TRANSD가 적절한건지 확인할 필요 있음.
nvmin, nvmax = priors["nvoro"]
rand_seed, rand_rmul = priors["rand"]
w0, w1, dw = beta


os.makedirs("./OUT", exist_ok=True)
logfile = "./OUT/Par.log"
log = open(logfile, "w")

def both(*args, **kwargs):
    print(*args, **kwargs)
    print(*args, **kwargs, file=log)

both("### PARAMETERS ###")
both(f"Ncore                    : {Ncore}")
both(f"Sampler                  : {sampler_type}")
both(f"Burnin                   : {burnin}")
both(f"Sample                   : {sample}")
both(f"Skip                     : {skip}")
both(f"No.Data                  : {Ndat}")
for i, ds in enumerate(datasets, 1):
    both(f"  [{i:02d}] {ds['type']}             : {ds['file']}")
    
both("\n### Priors ###")
both(f"Depth (zmin, zmax, dz)   : {zmin}, {zmax}, {dz}")
both(f"Vs (vsmin, vsmax, dvs)   : {vsmin}, {vsmax}, {dvs}")
both(f"Vs transD                : {vs_transd}")
both(f"No.Layer (nvmin, nvmax)  : {nvmin}, {nvmax}")
both(f"Rand (seed, rmul)        : {rand_seed}, {rand_rmul}")

both("\n### Weights ###")
both(f"w0, w1, dw               : {w0}, {w1}, {dw}")

log.close()


###################
# Custom prior
###################
if custum_prior:
    vs_prior = custom_prior_vs(zmin, zmax, vsmin, vsmax, vs_transd)
else:
    vs_prior = UniformPrior(
        name="vs",
        vmin=vsmin,
        vmax=vsmax,
        perturb_std=vs_transd,    # 여기 들어가는 변수 맞는지 다시 확인
        )

voronoi = Voronoi1D(
    name="voronoi",
    vmin=zmin,
    vmax=zmax,
    perturb_std=dz,
    n_dimensions=None,
    n_dimensions_min=int(nvmin),
    n_dimensions_max=int(nvmax),
    parameters=[vs_prior],
    )   

param_spaces = [voronoi]

for ds in datasets:
    name = ds["type"]   # e.g., RPV01, RGV01
    w_space = ParameterSpace(
        name=name,
        n_dimensions=1,
        parameters=[UniformPrior(name="w", vmin=w0, vmax=w1, perturb_std=dw)]
        )
    param_spaces.append(w_space)

parameterization = Parameterization(param_spaces)

if custom_logL:
    _Hi_loglike = partial(Hi_loglike, datasets=datasets)
    log_likelihood = LogLikelihood(log_like_func=_Hi_loglike)

else:
    targets, forwards = _get_target_forward(datasets, beta)
    log_likelihood = LogLikelihood(
        targets = targets,
        fwd_functions = forwards
        )

####################
# Sampler: Parallel Tempering or SimulatedAnnealing or None
####################
if sampler_type == "PT":
    if Ncore < 2:
        sampler = SimulatedAnnealing(
            temperature_start               = 10,
            cooling_fraction                = 0.8
            )
    else:
        sampler = ParallelTempering(
            temperature_max                 = 5,
            chains_with_unit_temperature    = 0.4,
            swap_every                      = 1000
            )

elif sampler_type == "SA":
    sampler = SimulatedAnnealing(
        temperature_start                   = 10,
        cooling_fraction                    = 0.8
        )

else:
    sampler = None
    
###################
# Inversion
###################
inversion = BayesianInversion(
    parameterization=parameterization,
    log_likelihood=log_likelihood,
    n_chains=Ncore,    
    )
inversion.run(
    n_iterations=Niter,
    burnin_iterations=burnin,
    save_every=skip,
    sampler = sampler,
    print_every= 1000, #max(Niter // 1000, 1)
    verbose=True,
    )



endtime = time.time()
print(f">>> End time   : {time.ctime(endtime)}")

duration = endtime - starttime
hours = int(duration // 3600)
minutes = int((duration % 3600) // 60)
seconds = int(duration % 60)
print(f">>> Duration   : {hours:02d}:{minutes:02d}:{seconds:02d}")


####################
# Save results
####################
import pickle, os
print(">>> DATA SAVING")
results = inversion.get_results(concatenate_chains=False)

os.makedirs("./OUT", exist_ok=True)
with open("./OUT/results.pkl", "wb") as f:
    pickle.dump(results, f)
# os.system(f"cp ./Par ./OUT/Par")
# 
