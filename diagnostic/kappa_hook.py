"""§7.3 κ_v 진단 훅 — HGAT/PyG 파이프라인용 (numpy 코어, torch 선택적)

목적: 층별 노드 임베딩 궤적에서 혼동 집합별 분리도 S_v와 붕괴 계수 κ_v를 계산해
스키마 링킹 오류(complete-recall 실패)와의 상관 분석(사전등록 P1·P2)을 가능하게 한다.

사용법 (서버):
    from kappa_hook import KappaDiagnostics
    # 혼동 집합: 동일 테이블 컬럼 묶음 (∪ 질의 후보 집합)
    diag = KappaDiagnostics(
        confusion_sets={"tbl_orders": [3, 4, 5, 6], "tbl_users": [9, 10, 11]},
        degrees={3: 1, 4: 2, ...},          # 선택: 닫힌형 κ_pred = 2 log(d+1) 비교용
    )
    diag.start_example("spider_dev_0017", extra={"gold": [4, 10], "recalled": [4]})
    diag.record(X0)                          # 입력 임베딩 [N, d]
    for H in layer_outputs:                  # 각 층 출력 [N, d] (torch.Tensor 가능)
        diag.record(H)
    rows = diag.finish_example()             # 노드별 κ_v 행 목록
    diag.to_csv("kappa_diag.csv")

torch 모델에 forward hook 자동 부착(선택):
    handles = diag.attach(model, module_filter=lambda name, m: name.startswith("convs."))
    ... forward ...
    diag.detach()
"""
import csv
import json
import numpy as np


def _to_np(H):
    if hasattr(H, "detach"):
        H = H.detach().cpu().numpy()
    return np.asarray(H, float)


class KappaDiagnostics:
    def __init__(self, confusion_sets, degrees=None, max_pairs=64, slope_lo=1):
        """confusion_sets: {set_id: [node_id, ...]} — 같은 집합 내 노드들이 서로의 혼동 후보.
        degrees: {node_id: degree} — 제공 시 닫힌형 예측 κ_pred = 2 log(d+1) 열 추가.
        max_pairs: 넓은 집합에서 분리도 계산에 쓸 상대 노드 수 상한.
        slope_lo: κ 회귀에서 건너뛸 초기 층 수(기본 1: 입력층 제외)."""
        self.sets = {k: list(v) for k, v in confusion_sets.items()}
        self.degrees = dict(degrees or {})
        self.max_pairs = int(max_pairs)
        self.slope_lo = int(slope_lo)
        self.rows = []
        self._layers = None
        self._handles = []
        self.ex_id, self.extra = None, {}

    # ---------- 기록 ----------
    def start_example(self, ex_id, extra=None):
        self.ex_id, self.extra = ex_id, dict(extra or {})
        self._layers = []

    def record(self, H):
        self._layers.append(_to_np(H))

    # ---------- 계산 ----------
    def _slope(self, S):
        s = np.asarray(S[self.slope_lo:], float)
        s = s[s > 1e-280]
        if len(s) < 2:
            return float("nan")
        return float(-np.polyfit(np.arange(len(s)), np.log(s), 1)[0])

    def _global_scale(self, T):
        """전역 RMS 제곱 쌍거리(샘플) — 상대 κ의 분모"""
        L1, N, _ = T.shape
        r = np.random.default_rng(0)
        idx = r.integers(0, N, size=(min(512, max(8, N * 2)), 2))
        idx = idx[idx[:, 0] != idx[:, 1]]
        return ((T[:, idx[:, 0], :] - T[:, idx[:, 1], :]) ** 2).sum(-1).mean(-1)

    def _compare_metrics(self, T, v, others):
        """§B.1 노드별 비교지표 (dirichlet_norm/unnorm, mad). N(v)=others(혼동집합),
        층(layer) 평균으로 노드당 1값. κ_rel과 동일 혼동집합 기준이라 공정 비교."""
        Tv = T[:, v, :]                                   # [L, d]
        To = T[:, others, :]                              # [L, m, d]
        diff = Tv[:, None, :] - To                        # [L, m, d]
        dir_unnorm = float((diff ** 2).sum(-1).sum(-1).mean())     # 이웃 합 → 층 평균
        dv = np.sqrt(self.degrees.get(v, 0) + 1.0)
        do = np.sqrt(np.array([self.degrees.get(j, 0) + 1.0 for j in others], float))  # [m]
        diffn = Tv[:, None, :] / dv - To / do[None, :, None]
        dir_norm = float((diffn ** 2).sum(-1).sum(-1).mean())
        vn = np.linalg.norm(Tv, axis=-1)                  # [L]
        on = np.linalg.norm(To, axis=-1)                  # [L, m]
        dot = (Tv[:, None, :] * To).sum(-1)               # [L, m]
        cos = dot / (vn[:, None] * on + 1e-9)             # [L, m]
        mad_val = float((1.0 - cos).mean())               # 이웃·층 평균
        return {"dirichlet_norm": dir_norm,
                "dirichlet_unnorm": dir_unnorm,
                "mad": mad_val}

    def finish_example(self):
        assert self._layers, "record()로 층별 임베딩을 먼저 기록하세요."
        T = np.stack(self._layers)            # [L+1, N, d]
        glob = self._global_scale(T)
        out = []
        for sid, nodes in self.sets.items():
            for v in nodes:
                others = [u for u in nodes if u != v][: self.max_pairs]
                if not others:
                    continue
                S = ((T[:, [v], :] - T[:, others, :]) ** 2).sum(-1).mean(-1)  # [L+1]
                row = dict(example_id=self.ex_id, node_id=v, set_id=sid,
                           set_size=len(nodes),
                           S_first=float(S[min(1, len(S) - 1)]), S_last=float(S[-1]),
                           kappa_abs=self._slope(S),
                           kappa_rel=self._slope(S / np.maximum(glob, 1e-300)))
                if v in self.degrees:
                    row["kappa_pred"] = float(2 * np.log(self.degrees[v] + 1))
                # §B.1 (work_instructions_P0) 비교지표 — 동일 N(v)=혼동집합, 층 평균.
                #   dirichlet_unnorm = Σ_{j∈N} |x_i - x_j|²
                #   dirichlet_norm   = Σ_{j∈N} |x_i/√(d_i+1) - x_j/√(d_j+1)|²  (정규화)
                #   mad              = mean_{j∈N} (1 - cos(x_i, x_j))           (Chen 2020)
                row.update(self._compare_metrics(T, v, others))
                row.update(self.extra)
                out.append(row)
        self.rows.extend(out)
        self._layers = None
        return out

    # ---------- torch hook (선택) ----------
    def attach(self, model, module_filter=None):
        def make_hook():
            def h(_mod, _inp, outp):
                if isinstance(outp, (tuple, list)):
                    outp = outp[0]
                self.record(outp)
            return h
        for name, mod in model.named_modules():
            if module_filter is None or module_filter(name, mod):
                self._handles.append(mod.register_forward_hook(make_hook()))
        return self._handles

    def detach(self):
        for h in self._handles:
            h.remove()
        self._handles = []

    # ---------- 출력 ----------
    def to_csv(self, path):
        if not self.rows:
            return
        keys = sorted({k for r in self.rows for k in r})
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            for r in self.rows:
                w.writerow(r)

    def to_json(self, path):
        with open(path, "w") as f:
            json.dump(self.rows, f, ensure_ascii=False, indent=1)