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
d_def = '/Users/seguuu/Project/02_Bayesian_inversion/SWDANI_SGmod/SWDANIpy/code_utils'
sys.path.append(f"{d_def}")
# from datacov import dcov
from read_par import read_par
from likelihood import Hi_loglike, _get_target_forward
# from init_param import init_voro
from custom_prior import custom_prior_vs

import time, os
starttime = time.time()

print(">>> Bayesbay version > 0.3.6")
print(f">>> Start time : {time.ctime(starttime)}")


custom_logL = False    
# Ref. "./code_utils/likelihood.py"
custum_vs_prior = True
# Ref. "./code_utils/custom_prior.py"



####################
# Read Par file
####################
par = read_par("./Par")
# Ref. "./code_utils/read_par.py"
# dict_keys(['Nchain', 'mcmc', 'Sampler', 'priors', 'weights', 'datasets'])

# par =  {
#     "Nchain": Ncore,
#     "mcmc": {"burnin": burnin, "sample": sample, "skip": skip},
#     "sampler": sampler,
#     "transD": {"vs": vs_transd, "xi": xi_transd, "vpvs": vpvs_transd, "rho":rho_transd},
#     "priors": {
#         "depth": (depth_min, depth_max, depth_delta),
#         "vs": (vs_min, vs_max, vs_delta),
#         "xi": (xi_min, xi_max, xi_delta, xi_sw, xi_transd),
#         "vpvs": (vpvs_min, vpvs_max, vpvs_delta, vpvs_sw, vpvs_transd),
#         "rho": (rho_min, rho_max, rho_delta, rho_sw, rho_transd),
#         "nvoro": (nvoro_min, nvoro_max),
#         "rand": (rand_seed, rand_rmul),
#         },
#     "weights": (w0, w1, dw),
#     "datasets": datasets,
#     }



####################
# parameterization
####################

burnin, sample, skip = int(par["mcmc"]["burnin"]), int(par["mcmc"]["sample"]), int(par["mcmc"]["skip"])
Niter = burnin + sample
Ncore = int(par["Nchain"])

priors = par["priors"]
transd = par["transD"]
beta   = par["weights"]
switch = par["switch"]
datasets = par["datasets"]

Ndat = int(len(datasets))

zmin, zmax, dz          = priors['depth']
vsmin, vsmax, dvs       = priors["vs"]
vpvsmin,vpvsmax, dvpvs  = priors["vpvs"]
rhomin, rhomax, drho    = priors["rho"]
nvmin, nvmax            = priors["nvoro"]
rand_seed, rand_rmul    = priors["rand"]
w0, w1, dw              = par["weights"]

vs_transd               = transd["vs"]         # ! 이 VS_TRANSD가 적절한건지 확인할 필요 있음.
vpvs_transd             = transd["vpvs"]
xi_transd               = transd["xi"]
rho_transd              = transd["rho"]

vpvs_sw                 = switch["vpvs"]
xi_sw                   = switch["xi"]
rho_sw                  = switch["rho"]

sampler_type            = par["sampler"]
# PT: Parallel Tempreing
# SA: Simulated Annealing
# None: None
if Ncore == 1 and sampler_type == "PT":
    print(f">>> Chains={Ncore}, Sampler={sampler_type}")
    print(">>> Parallel Tempering needs >1 chain.")
    print(">>> Switching to Simulated Annealing.")  


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
# both(f"Vs transD                : {vs_transd}")
both(f"No.Layer (nvmin, nvmax)  : {nvmin}, {nvmax}")
both(f"Rand (seed, rmul)        : {rand_seed}, {rand_rmul}")

both("\n### Weights ###")
both(f"w0, w1, dw               : {w0}, {w1}, {dw}")

log.close()


###################
# Custom prior
###################
priors = []
if custum_vs_prior:
    vs_prior = custom_prior_vs(zmin, zmax, vsmin, vsmax, dvs)
    priors.append(vs_prior)
else:
    vs_prior = UniformPrior(name="vs", vmin=vsmin, vmax=vsmax, perturb_std=dvs)
    priors.append(vs_prior)
    
if vpvs_sw == 0: 
    vpvs_prior = UniformPrior(name="vpvs", vmin=vpvsmin, vmax=vpvsmax, perturb_std=dvpvs)
    priors.append(vpvs_prior)

if rho_sw == 0: 
    rho_prior = UniformPrior(name="rho", vmin=rhomin, vmax=rhomax, perturb_std=drho)
    priors.append(rho_prior)
    

voronoi = Voronoi1D(
    name="voronoi",
    vmin=zmin,
    vmax=zmax,
    perturb_std=dz,
    n_dimensions=None,
    n_dimensions_min=int(nvmin),
    n_dimensions_max=int(nvmax),
    parameters=priors,
    )   

param_spaces = [voronoi]

# Prior range for DATA WEIGHT
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
