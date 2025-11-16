#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 12 18:54:15 2025

@author: segu
"""

## Bayesian inversion module
import bayesbay as bb
from bayesbay import BayesianInversion
from bayesbay.samplers import ParallelTempering, SimulatedAnnealing
from bayesbay.likelihood import LogLikelihood
from bayesbay.discretization import Voronoi1D
from bayesbay.prior import UniformPrior, CustomPrior
from bayesbay.parameterization import Parameterization, ParameterSpace

# Utils
import numpy as np
from functools import partial
import sys
d_def = '/Users/seguuu/Project/02_Bayesian_inversion/SWDANIpy/code_utils'
sys.path.append(f"{d_def}")
# from datacov import dcov
from read_par import read_par
from likelihood import Hi_loglike, _get_target_forward
from init_param import init_voro
from custom_prior import custom_prior_vs

####################
# Read Par file
####################
par = read_par("./Par")
# dict_keys(['Nchain', 'mcmc', 'priors', 'weights', 'datasets'])

Ncore = int(par["Nchain"])
# If Ncore = 1, Simulated Annealing (SA) will be applied
burnin, sample, skip = int(par["mcmc"]["burnin"]), int(par["mcmc"]["sample"]), int(par["mcmc"]["skip"])
transd = par["transD"]

priors = par["priors"]
beta = par["weights"]
datasets = par["datasets"]

Ndat = int(len(datasets))

sampler_type = "PT" #"PT"
# PT: Parallel Tempreing
# SA: Simulated Annealing
# None: None

custom_logL = True    
custum_prior = True

####################
# parameterization
####################

Niter = burnin + sample

zmin, zmax, dz = priors['depth']
vsmin, vsmax, dvs = priors["vs"]
vs_transd = transd["vs"]         # ! 이 VS_TRANSD가 적절한건지 확인할 필요 있음.
nvmin, nvmax = priors["nvoro"]
rand_seed, rand_rmul = priors["rand"]
w0, w1, dw = beta   # prior range / perturb 로 해석

    
###################
# Custom prior
###################
if custum_prior:
    vs_prior = custom_prior_vs(zmin, zmax, vsmin,vsmax, vs_transd)
else:
    vs_prior = UniformPrior(
        name="vs",
        vmin=vsmin,
        vmax=vsmax,
        perturb_std=vs_transd,    # 적당한 값. 원하면 par에서 따와도 됨
        )

voronoi = Voronoi1D(
    name="voronoi",
    vmin=zmin,
    vmax=zmax,
    perturb_std=dz,         # 층경계 흔들기 크기
    n_dimensions=None,
    n_dimensions_min=int(nvmin),
    n_dimensions_max=int(nvmax),
    parameters=[vs_prior],
    )   

param_spaces = [voronoi]

for ds in datasets:
    name = ds["type"]   # RPV01, RGV01, ...
    w_space = ParameterSpace(
        name=name,          # state[name]["w"] 와 매칭
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
        print(">>> Chain > 1 required for Parallel Tempering")
        print(">>> Instead Simulated Annealing Applied")
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


####################
# Save results
####################
import pickle, os
results = inversion.get_results(concatenate_chains=False)

os.makedirs("./OUT", exist_ok=True)
with open("./OUT/results.pkl", "wb") as f:
    pickle.dump(results, f)





#%%


    
    

# with open("./OUT/00_results.pkl", "rb") as f:
#     results = pickle.load(f)


# sampled_voronoi_nuclei = results['voronoi.discretization']
# sampled_thickness = [Voronoi1D.compute_cell_extents(n) for n in sampled_voronoi_nuclei]
# sampled_vs = results['voronoi.vs']
# interp_depths = np.linspace(zmin, zmax, 400)
# statistics_vs = Voronoi1D.get_tessellation_statistics(
#     sampled_thickness, sampled_vs, interp_depths, input_type='extents'
#     )

# fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6, 8), gridspec_kw={'width_ratios': [2.5, 1]})
# ax1, cbar = Voronoi1D.plot_tessellation_density(sampled_thickness, 
#                                                 sampled_vs, 
#                                                 input_type='extents', 
#                                                 ax=ax1, 
#                                                 cmap='binary', 
#                                                 vmin=zmin, 
#                                                 vmax=zmax)

# ax1.plot(statistics_vs['mean'], interp_depths, 'b', lw=2, label='Vs Ensemble Mean')
# ax1.plot(statistics_vs['median'], interp_depths, 'r', lw=2, label='Vs Ensemble Median')
# ax1.set_xlabel("Vs [km/s]")
# ax1.set_ylabel("Depth [km]")
# ax1.legend()

# Voronoi1D.plot_interface_hist(sampled_voronoi_nuclei, bins=75, ec='w', ax=ax2)
# ax2.tick_params(labelleft=False)
# ax2.set_ylabel('')
# ax2.set_ylim(*ax1.get_ylim())
# plt.tight_layout()
# plt.show()



# for ds in datasets:
#     name = ds["type"]              # 예: 'RPV01'
#     T = ds["t"]                    # periods
#     obs = ds["obs"]               # observed velocity
#     std = np.sqrt(ds["var"])      # standard deviation

#     key = f"{name}.dpred"         # 저장된 posterior predicted key
#     if key not in results:
#         print(f"{key} not in results")
#         continue

#     P = np.array(results[key])    # shape = (Nsamples, Nperiods)

#     # velocity range 자동 설정
#     vmin = np.nanpercentile(P, 1)
#     vmax = np.nanpercentile(P, 99)

#     # 히스토그램 bins
#     bins_vel = 150
#     vel_bins = np.linspace(vmin, vmax, bins_vel + 1)
#     vel_centers = 0.5 * (vel_bins[:-1] + vel_bins[1:])

#     # PPD 밀도 계산
#     density = np.zeros((bins_vel, len(T)))
#     for i in range(len(T)):
#         column = P[:, i]
#         column = column[np.isfinite(column)]
#         if len(column) > 0:
#             hist, _ = np.histogram(column, bins=vel_bins)
#             density[:, i] = hist

#     # normalize
#     if density.max() > 0:
#         density /= density.max()

#     # === Plot ===
#     fig, ax = plt.subplots(figsize=(6, 5))

#     # 배경: posterior predictive density
#     im = ax.pcolormesh(
#         T, vel_centers, density,
#         cmap='binary', shading='auto'
#         )
#     plt.colorbar(im, ax=ax, label="Normalized density")

#     ax.errorbar(
#         T, obs, yerr=std,
#         fmt='ro-', ms=4, capsize=2, label='Observed ±1σ'
#         )

#     ax.set_xlabel("Period (s)")
#     ax.set_ylabel("Velocity (km/s)")
#     ax.set_title(f"{name} Posterior Predictive Density")
#     ax.legend()

#     plt.tight_layout()
#     plt.show()
    
    
# # #%%
# # def state_from_results(results, k):
# #     """
# #     results dict에서 k번째 샘플을 BayesBay state dict 형태로 복원
# #     (voronoi + 각 데이터셋 weight w)
# #     """
# #     z_k = np.asarray(results["voronoi.discretization"][k], float)
# #     vs_k = np.asarray(results["voronoi.vs"][k], float)

# #     state = {
# #         "voronoi": {
# #             "discretization": z_k,
# #             "vs": vs_k,
# #         }
# #     }

# #     # 각 dataset의 w도 있으면 채워 넣기
# #     for key in results.keys():
# #         if key.endswith(".w"):
# #             # 예: "RPV01.w" -> name="RPV01"
# #             name = key.rsplit(".", 1)[0]
# #             w_k = np.asarray(results[key][k:k+1], float)  # shape (1,)
# #             state[name] = {"w": w_k}

# #     return state


# # def plot_pred_ppd_for_dataset(results, datasets, dataset_index=0,
# #                               Nsamp_plot=500, n_bins=60):
# #     """
# #     하나의 dataset (예: RPV01) 에 대해
# #     period vs predicted value PPD를 2D 이미지로 그림.
# #     """
# #     ds = datasets[dataset_index]
# #     t = ds["t"]
# #     obs = ds["obs"]
# #     forward = ds["forward"]

# #     Nsamp_total = results["voronoi.vs"].shape[0]
# #     Nsamp_plot = min(Nsamp_plot, Nsamp_total)

# #     # 균일한 간격으로 샘플 index 선택
# #     idx = np.linspace(0, Nsamp_total - 1, Nsamp_plot).astype(int)

# #     # forward 예측 모음
# #     dpred_list = []
# #     for k in idx:
# #         state = state_from_results(results, k)
# #         dpred = forward(state)
# #         if dpred.shape[0] != obs.shape[0]:
# #             continue
# #         dpred_list.append(dpred)

# #     dpred_samples = np.asarray(dpred_list)  # (Nsamp_eff, Ndata)
# #     Nsamp_eff, Ndata = dpred_samples.shape

# #     # 데이터 전체 범위
# #     vmin = float(dpred_samples.min())
# #     vmax = float(dpred_samples.max())
# #     v_bins = np.linspace(vmin, vmax, n_bins + 1)
# #     v_centers = 0.5 * (v_bins[:-1] + v_bins[1:])

# #     # period별 히스토그램 -> (Ndata, n_bins)
# #     pdf = np.zeros((Ndata, n_bins))
# #     for i in range(Ndata):
# #         hist, _ = np.histogram(dpred_samples[:, i], bins=v_bins, density=True)
# #         pdf[i, :] = hist

# #     # PPD 이미지
# #     plt.figure(figsize=(6, 4))
# #     extent = [v_centers[0], v_centers[-1], t[-1], t[0]]  # y축 뒤집기용
# #     plt.imshow(
# #         pdf,
# #         extent=extent,
# #         aspect="auto",
# #         origin="upper"
# #     )
# #     plt.colorbar(label="PDF")

# #     # 관측값 + errorbar
# #     std = np.sqrt(ds["var"])
# #     plt.errorbar(obs, t, xerr=std, fmt="w.", ms=3, capsize=2, label="obs")

# #     # median pred curve
# #     pred_p50 = np.percentile(dpred_samples, 50, axis=0)
# #     plt.plot(pred_p50, t, "w-", lw=2, label="median pred")

# #     plt.gca().invert_yaxis()
# #     plt.xlabel("Predicted value (e.g., velocity)")
# #     plt.ylabel("Period")
# #     plt.title(f"PPD of predicted data: {ds['type']}")
# #     plt.legend()
# #     plt.tight_layout()
# #     plt.show()


# # # 예: 첫 번째 dataset (index=0) 에 대해 PPD 그림
# # plot_pred_ppd_for_dataset(results, datasets, dataset_index=0)


    
# #     #%%
    
# # with open("./OUT/00_results.pkl", "rb") as f:
# #     results = pickle.load(f)
    

# # # 불러온 z, vs
# # z_samples = results["voronoi.discretization"]
# # vs_samples = results["voronoi.vs"]

# # # 그림용 depth grid
# # z_grid = np.linspace(0, 1, 200)   # 최종 최대 깊이에 맞게 수정

# # # 모든 샘플을 z_grid에 선형 보간
# # vs_grid = []
# # for z, vs in zip(z_samples, vs_samples):
# #     vs_grid.append(np.interp(z_grid, z, vs))

# # vs_grid = np.array(vs_grid)   # shape: (Nsample, Ngrid)

# # # 퍼센타일 계산
# # p50 = np.percentile(vs_grid, 50, axis=0)
# # p16 = np.percentile(vs_grid, 16, axis=0)
# # p84 = np.percentile(vs_grid, 84, axis=0)
    
# # plt.figure(figsize=(4, 6))

# # plt.plot(p50, z_grid, 'k', label="median")
# # plt.fill_betweenx(z_grid, p16, p84, color='gray', alpha=0.4, label="1σ")

# # plt.gca().invert_yaxis()
# # plt.yscale("log")
# # plt.xlabel("Vs (km/s)")
# # plt.ylabel("Depth (km)")
# # plt.title("Posterior Vs profile")
# # plt.show()


# # from likelihood import Hi_loglike   # forward를 포함

# # # 특정 샘플 state 만들기
# # def sample_to_state(k):
# #     z = z_samples[k]
# #     vs = vs_samples[k]
# #     state = {
# #         "voronoi": {"discretization": z, "vs": vs}
# #         }
# #     return state

# # state0 = sample_to_state(-1)  # 마지막 샘플

# # # forward curves
# # dpred = datasets[0]["forward"](state0)

# # # #%%
# # # def _compute_cell_extents_sorted(sites, zmin=0.0, zmax=None):
# # #     """Voronoi sites -> (정렬된 sites, thickness)"""
# # #     s = np.asarray(sites, float)
# # #     if s.size == 0:
# # #         return np.array([]), np.array([]), np.array([])

# # #     order = np.argsort(s)
# # #     s = s[order]
# # #     if zmax is None:
# # #         zmax = s.max()
# # #     mids = (s[:-1] + s[1:]) / 2.0
# # #     boundaries = np.concatenate(([zmin], mids, [zmax]))
# # #     thickness = np.diff(boundaries)
# # #     return s, order, thickness, boundaries


# # # def extract_vs_samples(inversion, zmin, zmax, nz=200):
# # #     """BayesBay 결과에서 Vs(z) 샘플 행렬 (nsample × nz) 생성"""
# # #     # 체인들을 하나로 붙여서 결과 얻기
# # #     try:
# # #         results = inversion.get_results(concatenate_chains=True)
# # #     except TypeError:
# # #         results = inversion.get_results()

# # #     # 키 자동 탐색 (버전마다 이름이 조금씩 다를 수 있으므로)
# # #     keys = list(results.keys())
# # #     k_sites = [k for k in keys if "discretization" in k][0]
# # #     k_vs    = [k for k in keys if k.endswith(".vs") or k.split(".")[-1] == "vs"][0]

# # #     sites_list = results[k_sites]   # 리스트 형태
# # #     vs_list    = results[k_vs]

# # #     nsample = len(vs_list)
# # #     z_grid = np.linspace(zmin, zmax, nz)

# # #     vs_samples = np.empty((nsample, nz), float)

# # #     for i, (sites, vs_cells) in enumerate(zip(sites_list, vs_list)):
# # #         sites = np.asarray(sites, float)
# # #         vs_cells = np.asarray(vs_cells, float)
# # #         s_sorted, order, thickness, boundaries = _compute_cell_extents_sorted(sites, zmin=zmin, zmax=zmax)
# # #         if s_sorted.size == 0:
# # #             vs_samples[i, :] = np.nan
# # #             continue
# # #         vs_sorted = vs_cells[order]
# # #         # 각 깊이 지점이 어느 셀에 속하는지 인덱스
# # #         idx = np.searchsorted(boundaries, z_grid, side="right") - 1
# # #         idx = np.clip(idx, 0, len(vs_sorted) - 1)
# # #         vs_samples[i, :] = vs_sorted[idx]

# # #     return z_grid, vs_samples


# # # # ==== 실제 사용 ====
# # # z_grid, vs_samples = extract_vs_samples(inversion, zmin, zmax, nz=200)

# # # # NaN 들어간 샘플 제외
# # # mask_good = np.all(np.isfinite(vs_samples), axis=1)
# # # vs_samples = vs_samples[mask_good]

# # # vs_mean = np.mean(vs_samples, axis=0)
# # # vs_p16  = np.percentile(vs_samples, 16, axis=0)
# # # vs_p84  = np.percentile(vs_samples, 84, axis=0)

# # # plt.figure(figsize=(4, 6))
# # # plt.fill_betweenx(z_grid, vs_p16, vs_p84, alpha=0.3, label='16–84%')
# # # plt.plot(vs_mean, z_grid, label='mean Vs')
# # # plt.gca().invert_yaxis()
# # # plt.xlabel("Vs (km/s)")
# # # plt.ylabel("Depth (km)")
# # # plt.legend()
# # # plt.title("Posterior Vs profile")
# # # plt.tight_layout()
# # # plt.show()

# # # #%%
# # # def plot_dispersion_fits(inversion, datasets, nsample_plot=100):
# # #     """
# # #     여러 데이터셋(RPV/LPV/RGV/LGV/ELL)에 대해
# # #     관측 vs posterior 예측을 플롯.
# # #     """
# # #     try:
# # #         results = inversion.get_results(concatenate_chains=True)
# # #     except TypeError:
# # #         results = inversion.get_results()

# # #     # state 재구성을 위해 결과 딕셔너리 확인
# # #     # BayesBay 결과 형식에 따라 약간 조정 필요할 수 있음.
# # #     keys = list(results.keys())
# # #     k_sites = [k for k in keys if "discretization" in k][0]
# # #     k_vs    = [k for k in keys if k.endswith(".vs") or k.split(".")[-1] == "vs"][0]

# # #     sites_list = results[k_sites]
# # #     vs_list    = results[k_vs]

# # #     nsample = len(vs_list)
# # #     idx_sample = np.linspace(0, nsample - 1, min(nsample_plot, nsample)).astype(int)

# # #     nrow = len(datasets)
# # #     plt.figure(figsize=(5, 3 * nrow))

# # #     for i_ds, ds in enumerate(datasets, start=1):
# # #         plt.subplot(nrow, 1, i_ds)
# # #         base = ds["type"][:3]
# # #         t = ds["t"]
# # #         obs = ds["obs"]
# # #         std = np.sqrt(ds["var"])

# # #         # 관측 에러바
# # #         plt.errorbar(1/t, obs, yerr=std, fmt='o', ms=3, capsize=2, label=f"{ds['type']} obs")

# # #         # posterior에서 몇 개 샘플 골라 예측 곡선
# # #         forward = ds["forward"]
# # #         for j in idx_sample:
# # #             # BayesBay의 state 포맷에 맞게 state 구성
# # #             state = {
# # #                 "voronoi": {
# # #                     "discretization": np.asarray(sites_list[j], float),
# # #                     "vs": np.asarray(vs_list[j], float),
# # #                 }
# # #             }
# # #             # 하이퍼파라미터 w는 여기선 평균 0 가정(혹은 나중에 결과에서 불러와도 됨)
# # #             # forward는 dpred만 계산
# # #             dpred = forward(state)
# # #             if dpred.shape[0] != obs.shape[0]:
# # #                 continue
# # #             if np.any(~np.isfinite(dpred)):
# # #                 continue
# # #             plt.plot(1/t, dpred, alpha=0.2)

# # #         plt.xlabel("Period (s)")
# # #         if base in ("RPV", "LPV", "RGV", "LGV"):
# # #             plt.ylabel("Velocity (km/s)")
# # #         elif base == "ELL":
# # #             plt.ylabel("Ellipticity")
# # #         plt.title(f"{ds['type']} fit")
# # #         # plt.legend()
# # #         plt.xscale('log')
# # #     plt.tight_layout()
# # #     plt.show()


# # # # ==== 실제 사용 ====
# # # plot_dispersion_fits(inversion, datasets, nsample_plot=100)

