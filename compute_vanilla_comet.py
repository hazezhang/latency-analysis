"""
计算Vanilla COMET-KIWI (Frozen)的Pearson相关系数
- 不训练，直接使用预训练的COMET模型
- 对每个segment: s = COMET(source, hypothesis) 得到单个scalar
- 分别计算: r(s, LQ) 和 r(s, EXP)
目的：看generic MT metric能否区分两个维度
"""
import json
import os
import numpy as np
from scipy.stats import pearsonr
from tqdm import tqdm

try:
    from comet import download_model, load_from_checkpoint
except ImportError:
    try:
        from unbabel_comet import download_model, load_from_checkpoint
    except ImportError:
        raise ImportError("Please install unbabel-comet: pip install unbabel-comet")


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
        return 0.0, "Prediction std is zero"
    if np.std(y) < 1e-8:
        return 0.0, "Target std is zero"
    try:
        corr, p_value = pearsonr(x, y)
        return corr, f"p-value: {p_value:.4f}"
    except Exception as e:
        return 0.0, f"Error: {str(e)}"


def main():
    test_data_path = "test_set.json"
    
    print("="*80)
    print("Vanilla COMET-KIWI (Frozen) Pearson Correlation")
    print("="*80)
    
    # Step 1: 加载COMET模型
    print("\nStep 1: Loading COMET model...")
    token_file = ".hf_token"
    hf_token = None
    if os.path.exists(token_file):
        with open(token_file, 'r') as f:
            hf_token = f.read().strip()
    if not hf_token:
        hf_token = os.getenv("HF_TOKEN")
    if hf_token:
        os.environ['HF_TOKEN'] = hf_token
    
    model_name = "Unbabel/wmt22-cometkiwi-da"
    model_path = download_model(model_name, saving_directory="./comet_models")
    comet_model = load_from_checkpoint(model_path)
    comet_model.eval()
    print("✅ COMET model loaded")
    
    # Step 2: 读取测试集
    print("\nStep 2: Loading test set...")
    with open(test_data_path, 'r', encoding='utf-8') as f:
        test_data = json.load(f)
    
    # Step 3: 对每个segment计算COMET分数
    print("\nStep 3: Computing COMET scores for each segment...")
    print("  s = COMET(source, hypothesis)")
    
    comet_scores = []
    test_lq_values = []
    test_exp_values = []
    valid_samples = []
    
    for item in tqdm(test_data, desc="Computing COMET scores"):
        src = item.get('src')
        mt = item.get('mt')
        lq = item.get('LQ')
        exp = item.get('EXP')
        
        if src is None or mt is None or lq is None or exp is None:
            continue
        
        try:
            # 计算COMET分数
            # COMET模型的predict方法接受(source, translation)对
            score = comet_model.predict([(src, mt)], batch_size=1)[0]
            # score可能是一个numpy array或scalar
            if isinstance(score, (list, np.ndarray)):
                score = float(score[0] if len(score) > 0 else score)
            else:
                score = float(score)
            
            # 量化标签
            lq_val = quantize_half(float(lq))
            exp_val = quantize_half(float(exp))
            
            comet_scores.append(score)
            test_lq_values.append(lq_val)
            test_exp_values.append(exp_val)
            valid_samples.append({
                'src': src[:50] + '...' if len(src) > 50 else src,
                'mt': mt[:50] + '...' if len(mt) > 50 else mt,
                'comet_score': score,
                'LQ': lq_val,
                'EXP': exp_val
            })
        except Exception as e:
            print(f"⚠️  Error processing sample: {e}")
            continue
    
    print(f"\nProcessed {len(comet_scores)} valid samples")
    
    # Step 4: 计算Pearson相关系数
    print("\nStep 4: Computing Pearson correlations...")
    print("-" * 80)
    
    # 转换为numpy数组
    comet_array = np.array(comet_scores)
    lq_array = np.array(test_lq_values)
    exp_array = np.array(test_exp_values)
    
    # 计算统计信息
    print(f"\nCOMET Score Statistics:")
    print(f"  Mean: {comet_array.mean():.4f}")
    print(f"  Std: {comet_array.std():.4f}")
    print(f"  Min: {comet_array.min():.4f}")
    print(f"  Max: {comet_array.max():.4f}")
    print(f"  Range: [{comet_array.min():.4f}, {comet_array.max():.4f}]")
    
    print(f"\nLQ Gold Statistics:")
    print(f"  Mean: {lq_array.mean():.4f}")
    print(f"  Std: {lq_array.std():.4f}")
    print(f"  Range: [{lq_array.min():.2f}, {lq_array.max():.2f}]")
    
    print(f"\nEXP Gold Statistics:")
    print(f"  Mean: {exp_array.mean():.4f}")
    print(f"  Std: {exp_array.std():.4f}")
    print(f"  Range: [{exp_array.min():.2f}, {exp_array.max():.2f}]")
    
    # 计算相关系数
    print("\n" + "="*80)
    print("Pearson Correlations:")
    print("="*80)
    
    # r(s, LQ)
    lq_corr, lq_info = safe_corr(comet_array, lq_array)
    print(f"\nr(s, LQ) = Pearson(COMET_score, LQ_gold):")
    print(f"  Correlation: {lq_corr:.6f}")
    print(f"  {lq_info}")
    
    # r(s, EXP)
    exp_corr, exp_info = safe_corr(comet_array, exp_array)
    print(f"\nr(s, EXP) = Pearson(COMET_score, EXP_gold):")
    print(f"  Correlation: {exp_corr:.6f}")
    print(f"  {exp_info}")
    
    # 总结
    print("\n" + "="*80)
    print("Summary:")
    print("="*80)
    print(f"Vanilla COMET-KIWI (Frozen):")
    print(f"  Model: {model_name}")
    print(f"  Training: None (frozen, no training)")
    print(f"  Prediction: s = COMET(source, hypothesis) - single scalar")
    print(f"\nPearson Correlations:")
    print(f"  r(s, LQ) = {lq_corr:.6f}")
    print(f"  r(s, EXP) = {exp_corr:.6f}")
    print("\nInterpretation:")
    print("  This shows how well a generic MT metric (COMET-KIWI)")
    print("  can distinguish between LQ and EXP dimensions.")
    print("  Higher correlation indicates better discrimination.")
    print("="*80 + "\n")
    
    # 保存结果
    results = {
        'model': model_name,
        'training': 'frozen (no training)',
        'prediction': 's = COMET(source, hypothesis)',
        'test_samples': len(comet_scores),
        'comet_score_stats': {
            'mean': float(comet_array.mean()),
            'std': float(comet_array.std()),
            'min': float(comet_array.min()),
            'max': float(comet_array.max())
        },
        'lq_pearson': float(lq_corr),
        'exp_pearson': float(exp_corr),
        'lq_info': lq_info,
        'exp_info': exp_info,
        'sample_predictions': valid_samples[:10]  # 保存前10个样本作为示例
    }
    
    with open('vanilla_comet_results.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print("✅ Results saved to vanilla_comet_results.json")
    
    # 打印前5个样本的详细预测
    print("\nSample Predictions (first 5):")
    print("-" * 80)
    for i, sample in enumerate(valid_samples[:5]):
        print(f"\nSample {i+1}:")
        print(f"  Source: {sample['src']}")
        print(f"  MT: {sample['mt']}")
        print(f"  COMET score: {sample['comet_score']:.4f}")
        print(f"  LQ gold: {sample['LQ']:.2f}")
        print(f"  EXP gold: {sample['EXP']:.2f}")


if __name__ == "__main__":
    main()
