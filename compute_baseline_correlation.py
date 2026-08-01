"""
计算Mean Baseline的Pearson相关系数（Sanity Check）
对所有测试样本预测：
  y^LQ = mean_train(LQ)
  y^EXP = mean_train(EXP)
然后计算：
  Pearson(pred_LQ, gold_LQ)
  Pearson(pred_EXP, gold_EXP)
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
        return 0.0, "Insufficient samples"
    if np.std(x) < 1e-8:
        return 0.0, "Prediction std is zero (constant predictions)"
    if np.std(y) < 1e-8:
        return 0.0, "Target std is zero (constant targets)"
    try:
        corr, p_value = pearsonr(x, y)
        return corr, f"p-value: {p_value:.4f}"
    except Exception as e:
        return 0.0, f"Error: {str(e)}"


def main():
    train_data_path = "train_set.json"
    test_data_path = "test_set.json"
    
    print("="*80)
    print("Mean Baseline Pearson Correlation (Sanity Check)")
    print("="*80)
    
    # Step 1: 计算训练集的均值
    print("\nStep 1: Computing training set means...")
    with open(train_data_path, 'r', encoding='utf-8') as f:
        train_data = json.load(f)
    
    train_lq_values = []
    train_exp_values = []
    
    for item in train_data:
        lq = item.get('LQ')
        exp = item.get('EXP')
        if lq is not None:
            try:
                lq_val = quantize_half(float(lq))
                train_lq_values.append(lq_val)
            except (ValueError, TypeError):
                pass
        if exp is not None:
            try:
                exp_val = quantize_half(float(exp))
                train_exp_values.append(exp_val)
            except (ValueError, TypeError):
                pass
    
    train_lq_mean = np.mean(train_lq_values)
    train_exp_mean = np.mean(train_exp_values)
    train_lq_std = np.std(train_lq_values)
    train_exp_std = np.std(train_exp_values)
    
    print(f"Training set statistics:")
    print(f"  LQ: mean = {train_lq_mean:.4f}, std = {train_lq_std:.4f}, n = {len(train_lq_values)}")
    print(f"  EXP: mean = {train_exp_mean:.4f}, std = {train_exp_std:.4f}, n = {len(train_exp_values)}")
    
    # Step 2: 读取测试集
    print("\nStep 2: Loading test set...")
    with open(test_data_path, 'r', encoding='utf-8') as f:
        test_data = json.load(f)
    
    # Step 3: 对所有测试样本进行预测（使用训练集均值）
    print("\nStep 3: Predicting for all test samples...")
    print(f"  y^LQ = mean_train(LQ) = {train_lq_mean:.4f}")
    print(f"  y^EXP = mean_train(EXP) = {train_exp_mean:.4f}")
    
    test_lq_gold = []
    test_exp_gold = []
    test_lq_pred = []  # 总是预测训练集均值
    test_exp_pred = []
    
    for item in test_data:
        lq = item.get('LQ')
        exp = item.get('EXP')
        if lq is not None and exp is not None:
            try:
                lq_val = quantize_half(float(lq))
                exp_val = quantize_half(float(exp))
                test_lq_gold.append(lq_val)
                test_exp_gold.append(exp_val)
                test_lq_pred.append(train_lq_mean)  # 预测值 = 训练集均值
                test_exp_pred.append(train_exp_mean)  # 预测值 = 训练集均值
            except (ValueError, TypeError):
                pass
    
    print(f"\nTest set samples: {len(test_lq_gold)}")
    
    # Step 4: 计算Pearson相关系数
    print("\nStep 4: Computing Pearson correlations...")
    print("-" * 80)
    
    # LQ相关系数
    lq_corr, lq_info = safe_corr(test_lq_pred, test_lq_gold)
    print(f"\nPearson(pred_LQ, gold_LQ):")
    print(f"  Correlation: {lq_corr:.6f}")
    print(f"  Note: {lq_info}")
    print(f"  Prediction values: all = {train_lq_mean:.4f} (constant)")
    print(f"  Gold values: mean = {np.mean(test_lq_gold):.4f}, std = {np.std(test_lq_gold):.4f}")
    print(f"  Gold range: [{np.min(test_lq_gold):.2f}, {np.max(test_lq_gold):.2f}]")
    
    # EXP相关系数
    exp_corr, exp_info = safe_corr(test_exp_pred, test_exp_gold)
    print(f"\nPearson(pred_EXP, gold_EXP):")
    print(f"  Correlation: {exp_corr:.6f}")
    print(f"  Note: {exp_info}")
    print(f"  Prediction values: all = {train_exp_mean:.4f} (constant)")
    print(f"  Gold values: mean = {np.mean(test_exp_gold):.4f}, std = {np.std(test_exp_gold):.4f}")
    print(f"  Gold range: [{np.min(test_exp_gold):.2f}, {np.max(test_exp_gold):.2f}]")
    
    # 总结
    print("\n" + "="*80)
    print("Summary:")
    print("="*80)
    print(f"Mean Baseline Predictions:")
    print(f"  y^LQ = {train_lq_mean:.4f} (constant)")
    print(f"  y^EXP = {train_exp_mean:.4f} (constant)")
    print(f"\nPearson Correlations:")
    print(f"  Pearson(pred_LQ, gold_LQ) = {lq_corr:.6f}")
    print(f"  Pearson(pred_EXP, gold_EXP) = {exp_corr:.6f}")
    print("\nNote: Since predictions are constant (no variance),")
    print("      the Pearson correlation should be 0.0 (or undefined).")
    print("      This is expected and serves as a sanity check baseline.")
    print("="*80 + "\n")
    
    # 保存结果
    results = {
        'train_lq_mean': float(train_lq_mean),
        'train_exp_mean': float(train_exp_mean),
        'test_samples': len(test_lq_gold),
        'lq_pearson': float(lq_corr),
        'exp_pearson': float(exp_corr),
        'lq_info': lq_info,
        'exp_info': exp_info
    }
    
    with open('baseline_correlation_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print("✅ Results saved to baseline_correlation_results.json")


if __name__ == "__main__":
    main()
