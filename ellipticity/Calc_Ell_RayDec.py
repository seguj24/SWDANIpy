#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 13 17:48:24 2025

@author: segu
"""
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from scipy.signal import cheb1ord, cheby1, lfilter, detrend


def raydecpy(vert, north, east, time,
                      fmin, fmax, fsteps,
                      cycles, dfpar, nwind):
    """
    MATLAB raydec1station.m 의 1:1 포트 버전 (수치 동작 최대한 동일하게 맞춤)

    입력:
        vert, north, east : 1D array (N,)
        time              : 1D array (N,), 균일 샘플링
        fmin, fmax        : 분석 주파수 범위 [Hz]
        fsteps            : 로그 스케일 주파수 샘플 개수
        cycles            : 스택 신호 길이 (주기 개수)
        dfpar             : 상대 대역폭 (보통 0.1)
        nwind             : 시간 윈도우 개수

    반환:
        V : (fsteps, nwind)  각 윈도우별 주파수 리스트 (MATLAB V=fl)
        W : (fsteps, nwind)  각 윈도우별 ellipticity (MATLAB W=el)
    """

    # ----- 입력을 column vector 형태로 가정 -----
    v1 = np.asarray(vert).ravel()
    n1 = np.asarray(north).ravel()
    e1 = np.asarray(east).ravel()
    t1 = np.asarray(time).ravel()

    if not (v1.shape == n1.shape == e1.shape == t1.shape):
        raise ValueError("vert, north, east, time 길이가 서로 같아야 합니다.")

    K0 = v1.size
    K = K0 // nwind
    tau = t1[1] - t1[0]
    DTmax = 30.0
    fnyq = 1.0 / (2.0 * tau)

    fstart = max(fmin, 1.0 / DTmax)
    fend = min(fmax, fnyq)

    fl = np.zeros((fsteps, nwind))
    el = np.zeros((fsteps, nwind))

    constlog = (fend / fstart) ** (1.0 / (fsteps - 1))

    # 윈도우 루프 (ind1 = 1..nwind)
    for ind1 in range(nwind):
        s0 = ind1 * K
        s1 = (ind1 + 1) * K

        vert_win = detrend(v1[s0:s1])
        north_win = detrend(n1[s0:s1])
        east_win = detrend(e1[s0:s1])
        time_win = t1[s0:s1]

        horizontalamp = np.zeros(fsteps)
        verticalamp = np.zeros(fsteps)

        Tmax = np.max(time_win)

        # (MATLAB: fl=fstart*constlog.^(cumsum(ones(fsteps,nwind))-1);)
        # 여기서는 윈도우마다 동일하게 직접 계산
        flist = np.zeros(fsteps)

        # 주파수 루프 (findex = 1..fsteps)
        for findex in range(fsteps):
            # 중심 주파수
            f = fstart * (constlog ** findex)
            flist[findex] = f
            fl[findex, ind1] = f

            # 필터 대역폭
            df = dfpar * f
            fmin_b = max(fstart, f - df / 2.0)
            fmax_b = min(fnyq, f + df / 2.0)

            DT = cycles / f
            wl = int(np.round(DT / tau))   # window length in samples

            # Chebyshev 필터 (MATLAB cheb1ord/cheby1와 동일 공식)
            passband = np.array([
                fmin_b + (fmax_b - fmin_b) / 10.0,
                fmax_b - (fmax_b - fmin_b) / 10.0
            ]) / fnyq
            stopband = np.array([
                fmin_b - (fmax_b - fmin_b) / 10.0,
                fmax_b + (fmax_b - fmin_b) / 10.0
            ]) / fnyq

            # 그대로 쓰면 0 미만/1 초과가 될 수 있으므로, MATLAB과 같이
            # 정상적인 주파수 범위가 들어온다고 가정
            na, wn = cheb1ord(passband, stopband, gpass=1, gstop=5)
            b, a = cheby1(na, 0.5, wn, btype='band')

            # taper (MATLAB의 0:1/round(N/100):1 과 동일하게 생성)
            N = time_win.size
            step = int(np.round(N / 100.0))
            if step < 1:
                step = 1

            taper1 = np.arange(0.0, 1.0 + 1e-12, 1.0 / step)
            if 2 * taper1.size > N:
                raise ValueError("데이터 길이가 너무 짧아서 MATLAB taper 공식을 그대로 쓸 수 없습니다.")
            taper2 = np.ones(N - 2 * taper1.size)
            taper3 = taper1[::-1]
            taper = np.concatenate([taper1, taper2, taper3])
            taper = taper.reshape(-1)

            # 필터링: MATLAB filter(ch1,ch2,...)와 1:1 대응 → lfilter 사용
            norths = lfilter(b, a, taper * north_win)
            easts = lfilter(b, a, taper * east_win)
            verts = lfilter(b, a, taper * vert_win)

            # 음→양 zero-crossing
            derive = (np.sign(verts[1:K]) - np.sign(verts[0:(K - 1)])) / 2.0

            vertsum = np.zeros(wl)
            horsum = np.zeros(wl)
            dvindex = 0

            # MATLAB: for index = ceil(1/(4*f*tau))+1 : length(derive)-wl
            start_M = int(np.ceil(1.0 / (4.0 * f * tau))) + 1   # 1-based
            end_M = len(derive) - wl                           # 1-based

            # 0-based 인덱스로 변환
            start_idx = start_M - 1
            end_idx_exclusive = end_M      # range(..., end_M) → 마지막 = end_M-1

            offset = int(np.floor(1.0 / (4.0 * f * tau)))

            for idx in range(start_idx, end_idx_exclusive):
                if derive[idx] == 1:
                    dvindex += 1
                    # vsig: vertical
                    vsig = verts[idx:idx + wl]

                    # esig / nsig: index-floor(1/(4*f*tau)) ~ + wl-1
                    i0 = idx - offset
                    esig = easts[i0:i0 + wl]
                    nsig = norths[i0:i0 + wl]

                    # MATLAB: integral1 = sum(vsig.*esig);
                    #         integral2 = sum(vsig.*nsig);
                    integral1 = np.sum(vsig * esig)
                    integral2 = np.sum(vsig * nsig)

                    # theta 계산 (atan → atan2 + 동일 보정)
                    if integral2 == 0:
                        # MATLAB에서도 NaN/Inf가 될 수 있는 경우라, 그냥 skip
                        continue
                    theta = np.arctan(integral1 / integral2)
                    if integral2 < 0:
                        theta += np.pi
                    theta = (theta + np.pi) % (2.0 * np.pi)

                    # radial 방향 성분
                    hsig = np.sin(theta) * esig + np.cos(theta) * nsig

                    # 상관계수
                    num = np.sum(vsig * hsig)
                    den = np.sqrt(np.sum(vsig ** 2) * np.sum(hsig ** 2))
                    if den == 0:
                        continue
                    correlation = num / den

                    if correlation >= -1:
                        vertsum = vertsum + (correlation ** 2) * vsig
                        horsum = horsum + (correlation ** 2) * hsig

            klimit = int(np.round(DT / tau))
            if klimit > wl:
                klimit = wl
            if klimit <= 0:
                verticalamp[findex] = 0.0
                horizontalamp[findex] = 0.0
            else:
                verticalamp[findex] = np.sqrt(np.sum(vertsum[:klimit] ** 2))
                horizontalamp[findex] = np.sqrt(np.sum(horsum[:klimit] ** 2))

        # 각 윈도우의 ellipticity
        with np.errstate(divide='ignore', invalid='ignore'):
            ellist = horizontalamp / verticalamp

        el[:, ind1] = ellist

    V = fl   # MATLAB V = fl
    W = el   # MATLAB W = el
    return V, W