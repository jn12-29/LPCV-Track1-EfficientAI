import numpy as np

def calculate_recall_at_k(similarity_scores, ground_truth_indices, k=10):
    """
    计算单个查询样本的 Recall@K
    
    参数:
    similarity_scores (list 或 np.array): 当前查询图像与所有候选文本的相似度得分。
    ground_truth_indices (list 或 set): 真实匹配文本在候选集中的索引列表。
    k (int): 评估前 K 个结果，默认为 10 (即 Recall@Top10)。
    
    返回:
    float: Recall@K 的得分。
    """
    # 边界情况：如果没有对应的真实文本，返回 0.0
    if not ground_truth_indices:
        return 0.0
        
    scores = np.array(similarity_scores)
    
    # 获取得分最高的 Top K 个文本的索引
    # np.argsort 返回从小到大的索引，[-k:] 取最大的 K 个，[::-1] 将其反转为从大到小
    top_k_indices = np.argsort(scores)[-k:][::-1]
    
    # 统计 Top K 中命中了多少个真实文本索引
    ground_truth_set = set(ground_truth_indices)
    hits = sum(1 for idx in top_k_indices if idx in ground_truth_set)
            
    # 计算 Recall 公式：Top K 中的命中数量 / 真实文本总数
    recall = hits / len(ground_truth_indices)
    
    return recall

# ==========================================
# 示例演示
# ==========================================
if __name__ == "__main__":
    # 假设候选池中共有 20 个文本
    num_candidates = 20
    
    # 设定：当前图片对应的正确描述文本在候选池中的索引为 2, 8, 14 (共 3 个 Ground-truth)
    ground_truths = [2, 8, 14]
    
    # 模拟模型输出的 20 个相似度得分 (0~1之间)
    np.random.seed(42) 
    sim_scores = np.random.rand(num_candidates)
    
    # 模拟模型预测情况：给索引 8 和 14 打了高分，但索引 2 打了低分
    sim_scores[8] = 0.95
    sim_scores[14] = 0.90
    sim_scores[2] = 0.15 
    
    # 执行计算
    recall_10 = calculate_recall_at_k(sim_scores, ground_truths, k=10)
    
    # 输出结果
    top_10_pred = np.argsort(sim_scores)[-10:][::-1]
    print(f"模型预测的 Top 10 索引: {top_10_pred}")
    print(f"真实正确答案的索引: {ground_truths}")
    print(f"Recall@Top10 计算结果: {recall_10:.2f} (即命中 2 个 / 总共 3 个)")