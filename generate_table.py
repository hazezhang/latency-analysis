"""
生成Table 1: Pearson Correlation on Test Set
"""
import json
import numpy as np
from scipy.stats import pearsonr

def quantize_half(x: float) -> float:
    """将值量化到 [0,3] 范围内的 0.5 step"""
    x = float(x)
    x = max(0.0, min(3.0, x))
    return round(x * 2) / 2.0


def safe_corr(x, y):
    """安全的相关系数计算"""
    if len(x) < 2 or len(y) < 2:
        return 0.0
    if np.std(x) < 1e-8 or np.std(y) < 1e-8:
        return 0.0
    try:
        return pearsonr(x, y)[0]
    except:
        return 0.0


def main():
    train_data_path = "train_set.json"
    test_data_path = "test_set.json"
    
    print("\n" + "="*80)
    print("Table 1: Pearson Correlation on Test Set")
    print("="*80)
    
    # 1. Mean baseline
    print("\n1. Computing Mean Baseline...")
    with open(train_data_path, 'r', encoding='utf-8') as f:
        train_data = json.load(f)
    
    train_lq_values = []
    train_exp_values = []
    for item in train_data:
        lq = item.get('LQ')
        exp = item.get('EXP')
        if lq is not None:
            try:
                train_lq_values.append(quantize_half(float(lq)))
            except:
                pass
        if exp is not None:
            try:
                train_exp_values.append(quantize_half(float(exp)))
            except:
                pass
    
    train_lq_mean = np.mean(train_lq_values)
    train_exp_mean = np.mean(train_exp_values)
    
    with open(test_data_path, 'r', encoding='utf-8') as f:
        test_data = json.load(f)
    
    test_lq_values = []
    test_exp_values = []
    test_lq_pred = []
    test_exp_pred = []
    
    for item in test_data:
        lq = item.get('LQ')
        exp = item.get('EXP')
        if lq is not None and exp is not None:
            try:
                lq_val = quantize_half(float(lq))
                exp_val = quantize_half(float(exp))
                test_lq_values.append(lq_val)
                test_exp_values.append(exp_val)
                test_lq_pred.append(train_lq_mean)
                test_exp_pred.append(train_exp_mean)
            except:
                pass
    
    baseline_lq_corr = safe_corr(test_lq_pred, test_lq_values)
    baseline_exp_corr = safe_corr(test_exp_pred, test_exp_values)
    
    # 2. Vanilla COMET-KIWI
    print("2. Computing Vanilla COMET-KIWI...")
    test_lq_comet = []
    test_exp_comet = []
    comet_scores = []
    
    for item in test_data:
        lq = item.get('LQ')
        exp = item.get('EXP')
        comet_score = item.get('COMETkiwi_score')
        if lq is not None and exp is not None and comet_score is not None:
            try:
                test_lq_comet.append(quantize_half(float(lq)))
                test_exp_comet.append(quantize_half(float(exp)))
                # COMET-KIWI分数映射到[0,3]
                comet_mapped = (float(comet_score) + 1) / 2 * 3
                comet_scores.append(comet_mapped)
            except:
                pass
    
    comet_lq_corr = safe_corr(comet_scores, test_lq_comet) if len(comet_scores) > 0 else 0.0
    comet_exp_corr = safe_corr(comet_scores, test_exp_comet) if len(comet_scores) > 0 else 0.0
    
    # 3-4. 其他模型变体（需要单独训练，这里使用TODO）
    # 5. Dual-head (使用用户提供的值)
    dual_lq_corr = 0.388
    dual_exp_corr = 0.301
    
    # 打印表格
    print("\n" + "="*80)
    print("Table 1: Pearson Correlation on Test Set")
    print("="*80)
    print(f"{'Model':<35} {'LQ r':<12} {'EXP r':<12}")
    print("-" * 80)
    print(f"{'Mean baseline':<35} {baseline_lq_corr:<12.3f} {baseline_exp_corr:<12.3f}")
    print(f"{'Vanilla COMET-KIWI':<35} {comet_lq_corr:<12.3f} {comet_exp_corr:<12.3f}")
    print(f"{'Frozen encoder + linear':<35} {'[TODO]':<12} {'[TODO]':<12}")
    print(f"{'Single-head fine-tune':<35} {'[TODO]':<12} {'[TODO]':<12}")
    print(f"{'Dual-head (proposed)':<35} {dual_lq_corr:<12.3f} {dual_exp_corr:<12.3f}")
    print("="*80)
    
    # 生成LaTeX表格格式
    print("\n" + "="*80)
    print("LaTeX Format:")
    print("="*80)
    print("\\begin{table}")
    print("\\centering")
    print("\\begin{tabular}{lcc}")
    print("\\toprule")
    print("Model & LQ $r$ & EXP $r$ \\\\")
    print("\\midrule")
    print(f"Mean baseline & {baseline_lq_corr:.3f} & {baseline_exp_corr:.3f} \\\\")
    print(f"Vanilla COMET-KIWI & {comet_lq_corr:.3f} & {comet_exp_corr:.3f} \\\\")
    print("Frozen encoder + linear & [TODO] & [TODO] \\\\")
    print("Single-head fine-tune & [TODO] & [TODO] \\\\")
    print(f"Dual-head (proposed) & {dual_lq_corr:.3f} & {dual_exp_corr:.3f} \\\\")
    print("\\bottomrule")
    print("\\end{tabular}")
    print("\\caption{Pearson correlation on the test set.}")
    print("\\label{tab:results}")
    print("\\end{table}")
    print("="*80 + "\n")
    
    # 保存结果
    results = {
        'mean_baseline': {'lq': baseline_lq_corr, 'exp': baseline_exp_corr},
        'vanilla_comet': {'lq': comet_lq_corr, 'exp': comet_exp_corr},
        'frozen_linear': {'lq': None, 'exp': None},
        'single_head': {'lq': None, 'exp': None},
        'dual_head': {'lq': dual_lq_corr, 'exp': dual_exp_corr}
    }
    
    with open('table1_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print("✅ Results saved to table1_results.json")


if __name__ == "__main__":
    main()
