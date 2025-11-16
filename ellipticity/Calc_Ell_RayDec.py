#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 13 17:48:24 2025

@author: segu
"""

import numpy as np
from scipy.signal import cheb1ord, cheby1, filtfilt, detrend


def raydec1station_py(vert, north, east, time,
                      fmin, fmax, fsteps,
                      cycles=10, dfpar=0.1, nwind=1):
    """
    파이썬 포트 버전의 raydec1station.

    Parameters
    ----------
    vert, north, east : 1D array
        수직 / 북-남 / 동-서 성분 (동일 길이)
    time : 1D array
        시간축 (초), 균일 샘플링 가정
    fmin, fmax : float
        분석 주파수 범위 [Hz]
    fsteps : int
        로그 스케일 상 주파수 샘플 개수
    cycles : float
        스택 신호 길이를 주기 단위로 지정 (MATLAB CYCLES)
    dfpar : float
        상대 대역폭 (MATLAB DFPAR, 보통 0.1)
    nwind : int
        전체 신호를 나눌 시간 윈도우 개수

    Returns
    -------
    freqs : (fsteps, nwind) array
        각 윈도우별 주파수 리스트
    ell   : (fsteps, nwind) array
        Rayleigh-wave ellipticity (H_radial / V)
    """

    # ---- 입력 형식 정리 (N x 1 형태로 가정) ----
    vert = np.asarray(vert).ravel()
    north = np.asarray(north).ravel()
    east = np.asarray(east).ravel()
    time = np.asarray(time).ravel()

    if not (vert.shape == north.shape == east.shape == time.shape):
        raise ValueError("vert, north, east, time 길이가 서로 같아야 합니다.")

    K0 = vert.size
    K = K0 // nwind          # 한 윈도우 길이
    tau = time[1] - time[0]  # 샘플 간격
    DTmax = 30.0
    fnyq = 1.0 / (2.0 * tau)

    # MATLAB 코드의 fstart / fend
    fstart = max(fmin, 1.0 / DTmax)
    fend = min(fmax, fnyq)

    # 로그 스케일 주파수 설정
    constlog = (fend / fstart) ** (1.0 / (fsteps - 1))
    fl = np.zeros((fsteps, nwind))
    el = np.zeros((fsteps, nwind))

    for iw in range(nwind):
        # ---- 윈도우별 데이터 추출 & detrend ----
        s = slice(iw * K, (iw + 1) * K)
        v = detrend(vert[s])
        n = detrend(north[s])
        e = detrend(east[s])
        t = time[s]

        horizontalamp = np.zeros(fsteps)
        verticalamp = np.zeros(fsteps)

        Tmax = t.max()

        for fi in range(fsteps):
            # 중심 주파수
            f = fstart * (constlog ** fi)

            # 필터 대역폭 설정
            df = dfpar * f
            fmin_b = max(fstart, f - df / 2.0)
            fmax_b = min(fnyq, f + df / 2.0)
            fl[fi, iw] = f

            # 스택 길이 (주기 기준 cycles)
            DT = cycles / f
            wl = int(round(DT / tau))  # window length in samples

            # Chebyshev bandpass 필터 설계 (MATLAB cheb1ord + cheby1 대응)
            # passband / stopband 경계
            # 주파수는 0~fnyq 사이 정규화 (scipy는 0~1에서 1이 Nyquist)
            bw = (fmax_b - fmin_b)
            wp = [fmin_b + bw / 10.0, fmax_b - bw / 10.0]
            ws = [max(1e-6, fmin_b - bw / 10.0), min(fnyq - 1e-6, fmax_b + bw / 10.0)]

            wp_n = np.array(wp) / fnyq
            ws_n = np.array(ws) / fnyq

            # MATLAB: cheb1ord(..., 1, 5)
            N, wn = cheb1ord(wp_n, ws_n, gpass=1, gstop=5)
            # MATLAB: cheby1(na, 0.5, wn)
            b, a = cheby1(N, rp=0.5, Wn=wn, btype='band')

            # taper (양끝 1% 정도 코사인 테이퍼 유사)
            nt = len(t)
            taper_len = max(1, nt // 100)
            taper1 = np.linspace(0, 1, taper_len, endpoint=False)
            taper3 = taper1[::-1]
            if 2 * taper_len >= nt:
                # 데이터가 너무 짧으면 그냥 전체 taper만 적용
                taper = np.hanning(nt)
            else:
                taper2 = np.ones(nt - 2 * taper_len)
                taper = np.concatenate([taper1, taper2, taper3])

            # 필터링 (위상 보존 위해 filtfilt 사용)
            vs = filtfilt(b, a, v * taper)
            ns = filtfilt(b, a, n * taper)
            es = filtfilt(b, a, e * taper)

            # 음→양 zero-crossing 찾기 (derive == 1)
            derive = (np.sign(vs[1:]) - np.sign(vs[:-1])) / 2.0

            vertsum = np.zeros(wl)
            horsum = np.zeros(wl)

            # MATLAB: index = ceil(1/(4*f*tau))+1 : length(derive)-wl
            offset = int(np.floor(1.0 / (4.0 * f * tau)))
            istart = int(np.ceil(1.0 / (4.0 * f * tau)))
            iend = len(derive) - wl

            if iend <= istart:
                continue  # 데이터 길이가 너무 짧은 경우 건너뛰기

            for idx in range(istart, iend):
                if derive[idx] == 1:
                    # vsig: vertical
                    vsig = vs[idx: idx + wl]

                    # esig / nsig: 약간 앞에서 시작 (offset)
                    i0 = idx - offset
                    if i0 < 0 or i0 + wl > nt:
                        continue
                    esig = es[i0: i0 + wl]
                    nsig = ns[i0: i0 + wl]

                    # azimuth 추정 (MATLAB: atan(integral1/integral2) + 보정)
                    integral1 = np.sum(vsig * esig)
                    integral2 = np.sum(vsig * nsig)

                    # atan2가 더 안전 (integral2=0 방지)
                    theta = np.arctan2(integral1, integral2)

                    if integral2 < 0:
                        theta += np.pi
                    theta = (theta + np.pi) % (2.0 * np.pi)

                    # radial 방향 수평 성분
                    hsig = np.sin(theta) * esig + np.cos(theta) * nsig

                    # 상관계수 (항상 음수에 가깝다고 가정, MATLAB과 동일 정의)
                    num = np.sum(vsig * hsig)
                    den = np.sqrt(np.sum(vsig ** 2) * np.sum(hsig ** 2))
                    if den == 0:
                        continue
                    correlation = num / den

                    if correlation >= -1:  # MATLAB if correlation>=-1
                        w_corr = correlation ** 2
                        vertsum += w_corr * vsig
                        horsum += w_corr * hsig

            klimit = min(wl, int(round(DT / tau)))
            if klimit <= 0:
                continue

            verticalamp[fi] = np.sqrt(np.sum(vertsum[:klimit] ** 2))
            horizontalamp[fi] = np.sqrt(np.sum(horsum[:klimit] ** 2))

        # 이 윈도우에서의 ellipticity (H_radial / V)
        with np.errstate(divide='ignore', invalid='ignore'):
            ellist = horizontalamp / verticalamp
            ellist[~np.isfinite(ellist)] = np.nan

        el[:, iw] = ellist

    return fl, el