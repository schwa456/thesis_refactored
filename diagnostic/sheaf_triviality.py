"""§5.2 시프 자명성 측정 프로토콜 — 구현 v0.1 (numpy 코어, torch 선택적)

배경: HGNN의 관계별 변환 W_τ는 셀룰러 시프의 제한 사상으로 해석된다(Bodnar et al. 2022).
시프가 자명($W_τ \approx cI$)에 가까울수록 조화 공간이 상수 단면으로 퇴화하여
타입 간/전역 구분이 붕괴한다. 본 모듈은 학습된 가중치의 "자명성"을 정량화한다.

지표:
  T(W)      자명성 거리 = min_c ||W - cI||_F / ||W||_F  (닫힌형: c* = tr(W)/d)
            랜덤 기준선 E[T] ≈ sqrt(1 - 1/d^2) ≈ 1, 완전 자명 = 0.
  align     두 관계 사상의 정렬도 1 - |<W1,W2>_F| / (||W1||·||W2||)
  holonomy  사이클을 따른 제한 사상 곱의 자명성 T(∏ W) — 시프 불변량.
            자명 시프 ⇒ 모든 사이클에서 0.

프로토콜(체크포인트 대상):
  1) epoch 그리드마다 state_dict 저장 → scan_state_dict로 관계 가중치 추출
  2) 관계별 T, s_max 시계열 기록 (가설 5.2a: weight decay 하에서 T 하강)
  3) 스키마 그래프 FK 사이클 샘플에서 holonomy 기록 (가설 5.2b)
  4) κ_v(쌍둥이 축)와 T(시프 축)는 정리 4에 의해 직교하는 별개 붕괴 축 —
     2차원 (κ_v, T) 평면에서 모델 상태를 진단한다.
"""
import numpy as np


def triviality(W):
    """T(W) = min_c ||W - cI||_F / ||W||_F 와 최적 c* = tr(W)/d (닫힌형).
    근거: span{I}로의 직교 사영."""
    W = np.asarray(W, float)
    d = W.shape[0]
    c = float(np.trace(W) / d)
    num = np.linalg.norm(W - c * np.eye(d))
    den = np.linalg.norm(W) + 1e-300
    return float(num / den), c


def pair_alignment(W1, W2):
    """1 - |<W1, W2>_F| / (||W1||·||W2||): 0이면 스칼라배 관계(공유 방향)."""
    W1, W2 = np.asarray(W1, float), np.asarray(W2, float)
    inner = abs(float(np.sum(W1 * W2)))
    return float(1 - inner / (np.linalg.norm(W1) * np.linalg.norm(W2) + 1e-300))


def holonomy_gap(maps):
    """사이클을 따라가는 제한 사상 곱의 자명성 T(∏ maps). 자명 시프 ⇒ 0."""
    P = np.eye(np.asarray(maps[0]).shape[0])
    for M in maps:
        P = np.asarray(M, float) @ P
    return triviality(P)[0]


def scan_state_dict(state_dict, patterns=("rel", "edge", "W_r", "lin_r", "self", "root")):
    """torch state_dict에서 정방 가중치 행렬을 패턴으로 추출해 T, c*, s_max 산출.
    반환: [{param, d, T, c_opt, s_max}, ...]  (CSV/JSON 기록용)"""
    rows = []
    for name, ten in state_dict.items():
        try:
            arr = ten.detach().cpu().numpy()
        except AttributeError:
            arr = np.asarray(ten)
        if arr.ndim == 2 and arr.shape[0] == arr.shape[1] and any(p in name for p in patterns):
            t, c = triviality(arr)
            rows.append({
                "param": name, "d": int(arr.shape[0]),
                "T": round(t, 4), "c_opt": round(c, 4),
                "s_max": round(float(np.linalg.svd(arr, compute_uv=False)[0]), 4),
            })
    return rows


def sheaf_laplacian(edges, dim):
    """edges: [(u, v, F_u, F_v), ...] — 노드 인덱스와 제한 사상.
    블록 시프 라플라시안 L_F 반환 (검증·데모용 소규모 구현)."""
    N = 1 + max(max(u, v) for u, v, _, _ in edges)
    LF = np.zeros((N * dim, N * dim))
    for u, v, Fu, Fv in edges:
        iu, iv = u * dim, v * dim
        LF[iu:iu + dim, iu:iu + dim] += Fu.T @ Fu
        LF[iv:iv + dim, iv:iv + dim] += Fv.T @ Fv
        LF[iu:iu + dim, iv:iv + dim] -= Fu.T @ Fv
        LF[iv:iv + dim, iu:iu + dim] -= Fv.T @ Fu
    return LF