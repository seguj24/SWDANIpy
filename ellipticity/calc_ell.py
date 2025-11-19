#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from obspy import read
import numpy as np
from pathlib import Path
import scipy.io as sio  # pip install scipy
import os
import subprocess
from glob import glob
import matplotlib.pyplot as plt

# ===== 경로/스테이션 설정 =====
raydec_dir = Path('/home/seguuu/Project/02_Bayesian_inversion/SWDANIpy/ellipticity/src/RayDec')

dproj = "/home/seguuu/Project/996_ShallowVSprofile_SKP100sites"

tsl = [ # all 100 stations
        '001.KS.IMWB',
        '002.KS.YOCB',
        '003.KS.HAMB',
        '004.KS.HGSA',
        '005.KS.HAWB', # MASW 50-3000
        '006.KS.DDCA',
        '007.KG.HKU',
        '008.KG.KNUC',
        '009.KS.SEO3',
        '010.KS.DGLA',
        '011.KG.HSB',
        '012.KS.PORA',
        '013.KS.GLSA',
        '014.KS.YSAB',
        '015.KS.DGJA',
        '016.KS.JNUA',
        '017.KS.YUGA',
        '018.KS.BUYB',
        '019.KS.ISGB',
        '020.KG.NPR',
        '021.KS.SMWA',
        '022.KS.GOCB',
        '023.KS.YEGA',
        '024.KS.JAGA',
        '025.KS.JEU2',
        '026.KS.MIYA',
        '027.KS.JNYA',
        '028.KS.MLGA',
        '029.KG.YSB',
        '030.KS.MSNA',
        '031.KS.KOJ2',
        '032.KS.NOSA',
        '033.KS.KMSB',
        '034.KS.LIWA',
        '035.KS.TEJ2',
        '036.KS.CHJ3',
        '037.KS.GACA',
        '038.KS.EURB',
        '039.KS.HACA',
        '040.KS.HCNA',
        '041.KS.CHRB',
        '042.KS.PUAA',
        '043.KS.NAWB',
        '044.KG.JRB',
        '045.KS.BUSA',
        '046.KS.UJNA',
        '047.KS.YODB',
        '048.KG.JJB', # mam 300-5000
        '049.KS.HAWA', 
        '050.KS.HANB', 
        '051.KS.JAHA',
        '052.KS.MANA',
        '053.KG.MUN',
        '054.KS.SIJA',
        '055.KS.NAJA',
        '056.KG.SIG',
        '057.KS.HMPA',
        '058.KS.GGDA',
        '059.KG.DUC',
        '060.KS.YCHB',
        '061.KS.CGDA',
        '062.KG.GKP2',
        '063.KS.YGAA',
        '064.KS.JGNA',
        '065.KS.GICA',
        '066.KS.CIGB',
        '067.KS.GUWB',
        '068.KS.GUMA',
        '069.KS.DNBA',
        '070.KS.HWSA',
        '071.KS.OKCB',
        '072.KS.SKC2',
        '073.KS.YGGA',
        '074.KS.NCNA',
        '075.KS.SESA',
        '076.KS.NAMB',
        '077.KS.SNGB',
        '078.KS.LMGA',
        '079.KS.PYCB',
        '080.KS.PGEA',
        '081.KS.SNDA',
        '082.KS.YEYB',
        '083.KS.JAEA',
        '084.KS.JKJA',
        '085.KS.CSOA',
        '086.KS.EUSB',
        '087.KS.BLLA',
        '088.KS.CGPA',
        '089.KS.GKSA',
        '090.KS.TAIA',
        '091.KS.JASA',
        '092.KS.DGHA',
        '093.KS.BOGA',
        '094.KS.SLSA',
        '095.KS.AGSA',
        '096.KS.YAGA',
        '097.KS.YAPA',
        '098.KS.BKWA',
        '099.KS.CGAA',
        '100.KS.DGY2',
        ]

for sta in tsl:
    dsta = f"{dproj}/{sta}"
    
# sta = "001.KS.IMWB"
# dsta = f"{dproj}/{sta}"
    
    dell = f"{dsta}/ellipticity"
    # os.system(f"rm -rf {dell}")
    
    os.makedirs(dell, exist_ok=True)
    
    fndist = f"{dsta}/dist_acc2gp.txt"
    dist = np.loadtxt(fndist, dtype=str)
    sngp0 = dist[0][0]   # 예: '1658'
    
    ddat = f"{dsta}/data/mam/raw"
    
    # ===== mseed 읽기 =====
    stz = read(f"{ddat}/*{sngp0}.*.Z.miniseed")
    stn = read(f"{ddat}/*{sngp0}.*.N.miniseed")
    ste = read(f"{ddat}/*{sngp0}.*.E.miniseed")
    
    trz, trn, tre = stz[0], stn[0], ste[0]
    
    # 샘플링 간격 확인
    dtz = trz.stats.delta
    dtn = trn.stats.delta
    dte = tre.stats.delta
    
    if not (dtz == dtn == dte):
        raise ValueError("Inconsistent sampling rate among components.")
    
    dt = dtz
    
    # 공통 구간으로 trim
    t0 = max(trz.stats.starttime, trn.stats.starttime, tre.stats.starttime)
    t1 = min(trz.stats.endtime,   trn.stats.endtime,   tre.stats.endtime)
    
    trz = trz.trim(t0, t1, pad=True, fill_value=0)
    trn = trn.trim(t0, t1, pad=True, fill_value=0)
    tre = tre.trim(t0, t1, pad=True, fill_value=0)
    
    # numpy 배열로 추출
    vert  = trz.data.astype(float)
    north = trn.data.astype(float)
    east  = tre.data.astype(float)
    
    N = len(vert)
    time = np.arange(N) * dt  # 초 단위 시간축
    # RayDec 파라미터
    fmin   = 2      # Hz
    fmax   = 100.0      # Hz
    fsteps = 50
    cycles = 10
    dfpar  = 0.1
    
    
    # ---- 10분(600초)을 한 윈도우로 사용 ----
    T_total = N * dt            # 전체 길이 [sec]
    T_win   = 10 * 60           # 10분 = 600초
    
    nwind = int(T_total // T_win)  # 10분 윈도우 개수 (버림)
    if nwind < 1:
        nwind = 1  # 데이터가 10분보다 짧으면 전체를 1윈도우로
    
    print(f"총 길이 = {T_total/60:.1f} min, 윈도우 개수 = {nwind}")
    
    # 참고: 실제 한 윈도우 길이(레이덕 내부에서 쓸 K 기준)
    T_win_eff = (N / nwind) * dt
    print(f"실제 한 윈도우 길이 ≈ {T_win_eff/60:.1f} min")
    # ===== MATLAB용 입력 .mat 저장 =====
    infile = Path(dell) / f"raydec_input_{sngp0}.mat"
    
    sio.savemat(infile, {
        'vert':   vert,
        'north':  north,
        'east':   east,
        'time':   time,
        'dt':     float(dt),
        'fmin':   float(fmin),
        'fmax':   float(fmax),
        'fsteps': float(fsteps),
        'cycles': float(cycles),
        'dfpar':  float(dfpar),
        'nwind':  float(nwind),
        })
    
    print("저장:", infile)
    
    # #%%
    
    # script_dir = Path(dproj)  # calc_ell.m 있는 곳

    # ## Run Raydec
    # raydec_cmd = [
    #     "matlab",
    #     "-nodisplay",
    #     "-nosplash",
    #     "-nodesktop",
    #     "-r",
    #     (
    #         f"addpath('{script_dir.as_posix()}');"
    #         f"addpath('{raydec_dir.as_posix()}');"
    #         f"calc_ell('{sta}', '{dproj}');"
    #         "exit;"
    #     ),
    # ]

    # print(f"[{sta}] Raydec 실행 중...")
    # result = subprocess.run(
    #     raydec_cmd,
    #     stdout=subprocess.PIPE,
    #     stderr=subprocess.PIPE,
    #     text=True,
    #     )    

    # print(result.stdout)
    # if result.returncode != 0:
    #     print(result.stderr)
    #     raise RuntimeError(f"[{sta}] Raydec 실행 실패 (returncode={result.returncode})")
    # print(f"[{sta}] Raydec 완료\n")
# #%%
#     dproj = "/Users/seguuu/Project/996_ShallowVSprofile_SKP100sites"

#     dsta = f"{dproj}/001.KS.IMWB"
#     dell = f"{dsta}/ellipticity"
#     fndats = glob(f"{dell}/raydec_output_*.mat")[0]
    
    
#     data = sio.loadmat(fndats)
#     # print(data.keys())
#     # print(data["V"])
#     # print(data["W"])
    
#     frqs = data["V"]
#     ells = data["W"]
    
#     plt.plot(frqs, ells)
#     plt.xscale("log")
