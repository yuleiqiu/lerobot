import torch
from modeling_act import ACTTemporalEnsembler

def test_ACTTemporalEnsembler():
    """
    测试函数：
    1. 实例化 ACTTemporalEnsembler（参数可自行调整）。
    2. 每次生成随机的 actions（形状为 (batch, chunk_size, action_dim)）。
    3. 调用 update，并打印输入、返回的 action 以及内部变量的状态变化。
    """
    temporal_ensemble_coeff = 0.01
    chunk_size = 4
    batch_size = 2
    action_dim = 3

    # 实例化
    ensembler = ACTTemporalEnsembler(temporal_ensemble_coeff, chunk_size)
    print("初始 ensemble_weights:\n", ensembler.ensemble_weights)
    print("初始 ensemble_weights_cumsum:\n", ensembler.ensemble_weights_cumsum)

    num_updates = 5  # 模拟进行 5 次 update 调用
    for i in range(num_updates):
        print(f"\n第 {i+1} 次 update 调用：")
        # 生成随机的 actions 张量，形状为 (batch_size, chunk_size, action_dim)
        actions = torch.randn(batch_size, chunk_size, action_dim)
        print("输入 actions:\n", actions.shape)
        output_action = ensembler.update(actions)
        print("返回的 action:\n", output_action.shape)
        # print("当前 ensembled_actions:\n", ensembler.ensembled_actions)
        print("当前 ensembled_actions_count:\n", ensembler.ensembled_actions_count)

if __name__ == "__main__":
    test_ACTTemporalEnsembler()