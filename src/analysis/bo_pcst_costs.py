"""
방안 B: Bayesian Optimization for PCST cost parameters.
AdaptivePCSTExtractor의 고정 cost (belongs_to, fk, macro)를
Dev set F1 기준으로 최적화한다.

Usage:
    cd /home/hyeonjin/thesis_refactored
    conda run -n base python scripts/bo_pcst_costs.py [--n-trials 50]
"""
import os
import sys
import json
import argparse
import numpy as np
import optuna

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from utils.evaluator import parse_sql_elements

# ─── 1. 데이터 로드 (한 번만) ─────────────────────────────────
DEV_JSON = "data/raw/BIRD_dev/dev.json"
SCORE_DIR = "outputs/experiments/experiment_b_combined"  # Ensemble + Adaptive PCST (base)
PRED_DIR = SCORE_DIR

def load_dev_data():
    """score_analysis와 gold labels를 query별로 로드"""
    with open(DEV_JSON, 'r') as f:
        dev = json.load(f)

    # gold labels per query
    gold_map = {}
    for item in dev:
        qid = item['question_id']
        gold_sql = item.get('SQL', item.get('query', ''))
        gold_tables, gold_cols = parse_sql_elements(gold_sql)
        gold_map[qid] = {
            'db_id': item['db_id'],
            'gold_tables': set(t.lower() for t in gold_tables),
            'gold_cols': set(c.lower() for c in gold_cols),
        }

    # score + graph per query (single-file format)
    queries = {}
    score_files = [f for f in os.listdir(SCORE_DIR) if f.startswith('score_analysis')]
    for sf in score_files:
        with open(os.path.join(SCORE_DIR, sf), 'r') as f:
            for line in f:
                rec = json.loads(line)
                qid = rec['query_id']
                if qid not in queries:
                    queries[qid] = {'nodes': [], 'scores': [], 'gold': []}
                queries[qid]['nodes'].append(rec['node_name'])
                queries[qid]['scores'].append(rec['score'])
                queries[qid]['gold'].append(rec['is_gold'])

    return gold_map, queries


def load_graph_edges():
    """각 DB의 graph edges와 edge_types를 로드 (pipeline을 거치지 않고 직접 빌드)"""
    from modules.builders.graph_builder import HeteroGraphBuilder
    builder = HeteroGraphBuilder()

    db_dir = "data/raw/BIRD_dev/dev_databases"
    with open(DEV_JSON, 'r') as f:
        dev = json.load(f)

    db_ids = set(item['db_id'] for item in dev)
    graph_cache = {}

    for db_id in db_ids:
        _, metadata = builder.build(db_id=db_id, db_dir=db_dir)
        graph_cache[db_id] = {
            'edges': metadata['edges'],
            'edge_types': metadata['edge_types'],
            'node_metadata': metadata['node_metadata'],
        }

    # query -> db_id 매핑
    qid_to_db = {item['question_id']: item['db_id'] for item in dev}
    return graph_cache, qid_to_db


# ─── 2. PCST 실행 함수 ──────────────────────────────────
try:
    import pcst_fast
except ImportError:
    pcst_fast = None


def run_pcst_with_params(scores_arr, edges, edge_types,
                         belongs_to_cost, fk_cost, macro_cost,
                         percentile=80.0, min_prize=3, max_prize=25):
    """주어진 cost 파라미터로 Adaptive PCST를 실행"""
    if pcst_fast is None:
        raise ImportError("pcst_fast not installed")
    if len(edges) == 0:
        return list(range(len(scores_arr)))

    # Adaptive threshold
    threshold = np.percentile(scores_arr, percentile)
    prize_count = np.sum(scores_arr > threshold)
    if prize_count < min_prize:
        sorted_s = np.sort(scores_arr)[::-1]
        threshold = sorted_s[min(min_prize - 1, len(sorted_s) - 1)]
    elif prize_count > max_prize:
        sorted_s = np.sort(scores_arr)[::-1]
        threshold = sorted_s[min(max_prize - 1, len(sorted_s) - 1)]

    prizes = np.maximum(scores_arr - threshold, 0.0)

    cost_map = {
        'belongs_to': belongs_to_cost,
        'is_source_of': fk_cost,
        'points_to': fk_cost,
        'table_to_table': macro_cost,
    }
    costs = np.array([cost_map.get(et, 0.05) for et in edge_types], dtype=np.float64)
    edges_arr = np.array(edges, dtype=np.int64)

    selected_nodes, _ = pcst_fast.pcst_fast(edges_arr, prizes, costs, -1, 1, 'gw', 0)
    return selected_nodes.tolist()


def run_score_driven_pcst(scores_arr, edges, edge_types,
                          belongs_to_weight, fk_weight, macro_weight, epsilon,
                          percentile=80.0, min_prize=3, max_prize=25):
    """방안 A: Score-Driven cost로 PCST 실행"""
    if pcst_fast is None:
        raise ImportError("pcst_fast not installed")
    if len(edges) == 0:
        return list(range(len(scores_arr)))

    # Adaptive threshold
    threshold = np.percentile(scores_arr, percentile)
    prize_count = np.sum(scores_arr > threshold)
    if prize_count < min_prize:
        sorted_s = np.sort(scores_arr)[::-1]
        threshold = sorted_s[min(min_prize - 1, len(sorted_s) - 1)]
    elif prize_count > max_prize:
        sorted_s = np.sort(scores_arr)[::-1]
        threshold = sorted_s[min(max_prize - 1, len(sorted_s) - 1)]

    prizes = np.maximum(scores_arr - threshold, 0.0)

    weight_map = {
        'belongs_to': belongs_to_weight,
        'is_source_of': fk_weight,
        'points_to': fk_weight,
        'table_to_table': macro_weight,
    }
    costs = np.empty(len(edges), dtype=np.float64)
    for i, (u, v) in enumerate(edges):
        et = edge_types[i] if i < len(edge_types) else 'default'
        w = weight_map.get(et, 1.0)
        score_v = scores_arr[v] if v < len(scores_arr) else 0.0
        costs[i] = w * max(threshold - score_v, epsilon)

    edges_arr = np.array(edges, dtype=np.int64)
    selected_nodes, _ = pcst_fast.pcst_fast(edges_arr, prizes, costs, -1, 1, 'gw', 0)
    return selected_nodes.tolist()


def run_prize_relative_pcst(scores_arr, edges, edge_types,
                            bt_ratio, fk_ratio, macro_ratio,
                            percentile=80.0, min_prize=3, max_prize=25):
    """Prize 분포 기반 cost: cost = ratio × median(positive_prizes)"""
    if pcst_fast is None:
        raise ImportError("pcst_fast not installed")
    if len(edges) == 0:
        return list(range(len(scores_arr)))

    # Adaptive threshold
    threshold = np.percentile(scores_arr, percentile)
    prize_count = np.sum(scores_arr > threshold)
    if prize_count < min_prize:
        sorted_s = np.sort(scores_arr)[::-1]
        threshold = sorted_s[min(min_prize - 1, len(sorted_s) - 1)]
    elif prize_count > max_prize:
        sorted_s = np.sort(scores_arr)[::-1]
        threshold = sorted_s[min(max_prize - 1, len(sorted_s) - 1)]

    prizes = np.maximum(scores_arr - threshold, 0.0)

    # Prize 분포 기반 cost 결정
    positive_prizes = prizes[prizes > 0]
    if len(positive_prizes) > 0:
        median_prize = np.median(positive_prizes)
    else:
        median_prize = 0.01  # fallback

    cost_map = {
        'belongs_to': bt_ratio * median_prize,
        'is_source_of': fk_ratio * median_prize,
        'points_to': fk_ratio * median_prize,
        'table_to_table': macro_ratio * median_prize,
    }
    default_cost = median_prize
    costs = np.array([cost_map.get(et, default_cost) for et in edge_types], dtype=np.float64)
    edges_arr = np.array(edges, dtype=np.int64)

    selected_nodes, _ = pcst_fast.pcst_fast(edges_arr, prizes, costs, -1, 1, 'gw', 0)
    return selected_nodes.tolist()


# ─── 3. 평가 함수 ───────────────────────────────────────
def evaluate_params(gold_map, queries, graph_cache, qid_to_db,
                    cost_params, mode='fixed'):
    """전체 dev set에서 Recall/Precision/F1 계산"""
    total_tp, total_fp, total_fn = 0, 0, 0

    for qid, qdata in queries.items():
        if qid not in gold_map or qid not in qid_to_db:
            continue

        db_id = qid_to_db[qid]
        if db_id not in graph_cache:
            continue

        ginfo = graph_cache[db_id]
        scores_arr = np.array(qdata['scores'], dtype=np.float64)

        if mode == 'fixed':
            selected = run_pcst_with_params(
                scores_arr, ginfo['edges'], ginfo['edge_types'],
                cost_params['belongs_to'], cost_params['fk'], cost_params['macro']
            )
        elif mode == 'score_driven':
            selected = run_score_driven_pcst(
                scores_arr, ginfo['edges'], ginfo['edge_types'],
                cost_params['belongs_to_weight'], cost_params['fk_weight'],
                cost_params['macro_weight'], cost_params.get('epsilon', 1e-4)
            )
        elif mode == 'prize_relative':
            selected = run_prize_relative_pcst(
                scores_arr, ginfo['edges'], ginfo['edge_types'],
                cost_params['bt_ratio'], cost_params['fk_ratio'],
                cost_params['macro_ratio']
            )

        # Selected node names
        node_meta = ginfo['node_metadata']
        pred_set = set()
        for idx in selected:
            name = node_meta.get(idx, str(idx)).lower()
            if '->' in name:
                continue
            pred_set.add(name)

        gold_info = gold_map[qid]
        gold_set = set()
        for t in gold_info['gold_tables']:
            gold_set.add(t)
        for c in gold_info['gold_cols']:
            gold_set.add(c)

        tp = len(pred_set & gold_set)
        fp = len(pred_set - gold_set)
        fn = len(gold_set - pred_set)
        total_tp += tp
        total_fp += fp
        total_fn += fn

    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    return {'precision': precision, 'recall': recall, 'f1': f1,
            'tp': total_tp, 'fp': total_fp, 'fn': total_fn}


# ─── 4. Optuna Objectives ────────────────────────────────
def create_fixed_cost_objective(gold_map, queries, graph_cache, qid_to_db):
    def objective(trial):
        params = {
            'belongs_to': trial.suggest_float('belongs_to', 0.001, 0.2, log=True),
            'fk': trial.suggest_float('fk', 0.005, 0.5, log=True),
            'macro': trial.suggest_float('macro', 0.01, 2.0, log=True),
        }
        result = evaluate_params(gold_map, queries, graph_cache, qid_to_db, params, mode='fixed')
        # Log intermediate
        trial.set_user_attr('precision', result['precision'])
        trial.set_user_attr('recall', result['recall'])
        return result['f1']
    return objective


def create_prize_relative_objective(gold_map, queries, graph_cache, qid_to_db):
    def objective(trial):
        params = {
            'bt_ratio': trial.suggest_float('bt_ratio', 0.01, 1.0, log=True),
            'fk_ratio': trial.suggest_float('fk_ratio', 0.05, 2.0, log=True),
            'macro_ratio': trial.suggest_float('macro_ratio', 0.5, 10.0, log=True),
        }
        result = evaluate_params(gold_map, queries, graph_cache, qid_to_db, params, mode='prize_relative')
        trial.set_user_attr('precision', result['precision'])
        trial.set_user_attr('recall', result['recall'])
        return result['f1']
    return objective


def create_score_driven_objective(gold_map, queries, graph_cache, qid_to_db):
    def objective(trial):
        params = {
            'belongs_to_weight': trial.suggest_float('belongs_to_weight', 0.05, 2.0, log=True),
            'fk_weight': trial.suggest_float('fk_weight', 0.1, 3.0, log=True),
            'macro_weight': trial.suggest_float('macro_weight', 0.3, 10.0, log=True),
            'epsilon': trial.suggest_float('epsilon', 1e-5, 0.01, log=True),
        }
        result = evaluate_params(gold_map, queries, graph_cache, qid_to_db, params, mode='score_driven')
        trial.set_user_attr('precision', result['precision'])
        trial.set_user_attr('recall', result['recall'])
        return result['f1']
    return objective


# ─── 5. Main ─────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n-trials', type=int, default=50)
    args = parser.parse_args()

    print("Loading dev data...")
    gold_map, queries = load_dev_data()
    print(f"Loaded {len(queries)} queries")

    print("Building graph edges...")
    graph_cache, qid_to_db = load_graph_edges()
    print(f"Built graphs for {len(graph_cache)} databases")

    # ── Baseline: 현재 고정 cost ──
    print("\n" + "="*60)
    print("Baseline (current fixed costs)")
    print("="*60)
    baseline_params = {'belongs_to': 0.01, 'fk': 0.05, 'macro': 0.5}
    baseline = evaluate_params(gold_map, queries, graph_cache, qid_to_db, baseline_params, mode='fixed')
    print(f"  R={baseline['recall']:.4f} P={baseline['precision']:.4f} F1={baseline['f1']:.4f}")

    # ── Study 1: Fixed cost BO ──
    print("\n" + "="*60)
    print(f"Study 1: Bayesian Optimization of FIXED costs ({args.n_trials} trials)")
    print("="*60)
    study_fixed = optuna.create_study(direction='maximize', study_name='fixed_cost_bo')
    study_fixed.optimize(
        create_fixed_cost_objective(gold_map, queries, graph_cache, qid_to_db),
        n_trials=args.n_trials, show_progress_bar=True
    )
    best_fixed = study_fixed.best_trial
    print(f"\nBest Fixed Cost:")
    print(f"  belongs_to={best_fixed.params['belongs_to']:.4f}, "
          f"fk={best_fixed.params['fk']:.4f}, "
          f"macro={best_fixed.params['macro']:.4f}")
    print(f"  R={best_fixed.user_attrs['recall']:.4f} "
          f"P={best_fixed.user_attrs['precision']:.4f} "
          f"F1={best_fixed.value:.4f}")

    # ── Study 2: Prize-Relative BO ──
    print("\n" + "="*60)
    print(f"Study 2: Bayesian Optimization of PRIZE-RELATIVE ratios ({args.n_trials} trials)")
    print("="*60)
    study_pr = optuna.create_study(direction='maximize', study_name='prize_relative_bo')
    study_pr.optimize(
        create_prize_relative_objective(gold_map, queries, graph_cache, qid_to_db),
        n_trials=args.n_trials, show_progress_bar=True
    )
    best_pr = study_pr.best_trial
    print(f"\nBest Prize-Relative Ratios:")
    print(f"  bt_ratio={best_pr.params['bt_ratio']:.4f}, "
          f"fk_ratio={best_pr.params['fk_ratio']:.4f}, "
          f"macro_ratio={best_pr.params['macro_ratio']:.4f}")
    print(f"  R={best_pr.user_attrs['recall']:.4f} "
          f"P={best_pr.user_attrs['precision']:.4f} "
          f"F1={best_pr.value:.4f}")

    # ── Study 3: Score-Driven weight BO ──
    print("\n" + "="*60)
    print(f"Study 3: Bayesian Optimization of SCORE-DRIVEN weights ({args.n_trials} trials)")
    print("="*60)
    study_sd = optuna.create_study(direction='maximize', study_name='score_driven_bo')
    study_sd.optimize(
        create_score_driven_objective(gold_map, queries, graph_cache, qid_to_db),
        n_trials=args.n_trials, show_progress_bar=True
    )
    best_sd = study_sd.best_trial
    print(f"\nBest Score-Driven Weights:")
    print(f"  belongs_to_weight={best_sd.params['belongs_to_weight']:.4f}, "
          f"fk_weight={best_sd.params['fk_weight']:.4f}, "
          f"macro_weight={best_sd.params['macro_weight']:.4f}, "
          f"epsilon={best_sd.params['epsilon']:.6f}")
    print(f"  R={best_sd.user_attrs['recall']:.4f} "
          f"P={best_sd.user_attrs['precision']:.4f} "
          f"F1={best_sd.value:.4f}")

    # ── Summary ──
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"{'Method':<25} {'Recall':>8} {'Precision':>10} {'F1':>8}")
    print("-"*55)
    print(f"{'Baseline (fixed)':<25} {baseline['recall']:>8.4f} {baseline['precision']:>10.4f} {baseline['f1']:>8.4f}")
    print(f"{'BO Fixed Cost':<25} {best_fixed.user_attrs['recall']:>8.4f} {best_fixed.user_attrs['precision']:>10.4f} {best_fixed.value:>8.4f}")
    print(f"{'BO Prize-Relative':<25} {best_pr.user_attrs['recall']:>8.4f} {best_pr.user_attrs['precision']:>10.4f} {best_pr.value:>8.4f}")
    print(f"{'BO Score-Driven':<25} {best_sd.user_attrs['recall']:>8.4f} {best_sd.user_attrs['precision']:>10.4f} {best_sd.value:>8.4f}")

    # Save results
    results = {
        'baseline': {**baseline_params, **baseline},
        'bo_fixed': {**best_fixed.params, 'f1': best_fixed.value,
                     'recall': best_fixed.user_attrs['recall'],
                     'precision': best_fixed.user_attrs['precision']},
        'bo_prize_relative': {**best_pr.params, 'f1': best_pr.value,
                              'recall': best_pr.user_attrs['recall'],
                              'precision': best_pr.user_attrs['precision']},
        'bo_score_driven': {**best_sd.params, 'f1': best_sd.value,
                            'recall': best_sd.user_attrs['recall'],
                            'precision': best_sd.user_attrs['precision']},
    }
    out_path = 'notebooks/analysis_results/bo_pcst_cost_results.json'
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == '__main__':
    main()
