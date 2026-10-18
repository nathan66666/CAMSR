# analyze_casg.py
import torch
import os.path as osp
from basicsr.models import build_model
from basicsr.utils.options import parse_options

def analyze_casg_statistics(root_path):
    """分析训练好的CASG统计信息"""
    
    # 解析配置
    opt, args = parse_options(root_path, is_train=False)
    opt['root_path'] = root_path
    
    # 构建模型（加载训练好的权重）
    model = build_model(opt)
    # print(model)
    
    # 加载最新的模型
    # latest_model_path = osp.join(opt['path']['models'], 'latest.pth')
    latest_model_path = '/data5/CAMSR/experiments/test/models/net_g_200.pth'
    if osp.exists(latest_model_path):
        model.load_network(model.net_g, latest_model_path)
        print(f"Loaded model from {latest_model_path}")
        # print(model.net_g)
        # # CASG (TAB) 的完整路径
        # model.net_g.layers[0].residual_group.blocks[0].assm.semantic
    
    # 检查是否有CASG模块
    # for attr in dir(model.net_g.layers):
    #         if not attr.startswith('_'):
    #             print(f"  - {attr}")
    # print(model.net_g.layers[0].residual_group.blocks[0].assm.semantic )
    # if hasattr(model.net_g.layers[0].residual_group.blocks[0].assm.semantic, 'TAB') and model.net_g.layers[0].residual_group.blocks[0].assm.semantic.TAB.record_stats:
    #     casg = model.net_g.layers[0].residual_group.blocks[0].assm.semantic.TAB
    # 正确：semantic 本身就是 TAB (CASG) 模块
    casg = model.net_g.layers[0].residual_group.blocks[0].assm.semantic
    # casg.record_stats = True
    # casg.reset_stats()
    # print(f"✅ CASG statistics enabled: {casg.record_stats}")
    # 检查 history 是否真的有数据
    if hasattr(casg, 'cluster_size_history'):
        print(f"7. cluster_size_history type = {type(casg.cluster_size_history)}")
        if len(casg.cluster_size_history) > 0:
            print(f"8. First record = {casg.cluster_size_history[0]}")
        else:
            print(f"8. cluster_size_history is EMPTY")
    else:
        print(f"7. cluster_size_history attribute does NOT exist")

    # 直接检查 casg 的属性
    if casg.record_stats:
        print("record_stats is enabled")
        # 使用 casg 的方法
        # summary = casg.get_statistics_summary()
        # casg.plot_statistics()
        # 打印摘要
        print(casg.get_statistics_summary())
        summary = casg.get_statistics_summary()
        if summary:
            print("\n" + "=" * 60)
            print("CASG Cluster Statistics Summary")
            print("=" * 60)
            print(f"Total steps recorded: {summary['num_steps']}")
            print(f"Final cluster distribution: {summary['final_distribution']}")
            print(f"Mean over time: {summary['mean_over_time']}")
            print(f"Std over time: {summary['std_over_time']}")
            print(f"Min over time: {summary['min_over_time']}")
            print(f"Max over time: {summary['max_over_time']}")
            print(f"Average max/min ratio: {summary['avg_ratio']:.2f}")
            print(f"Empty clusters occurred: {summary['has_empty_clusters']}")
            print("=" * 60)
            
            # 生成统计图
            save_path = osp.join(opt['path']['experiments_root'], 'casg_cluster_analysis.pdf')
            print(f"Saving statistics plot to: {save_path}")  # 添加这一行
            casg.plot_statistics(save_path=save_path)
            print(f"\nStatistics plot saved to {save_path}")
        else:
            print("No statistics recorded. Make sure record_stats=True during training.")
    else:
        print("No CASG module found or record_stats is disabled.")

if __name__ == '__main__':
    root_path = osp.abspath(osp.join(__file__, osp.pardir, osp.pardir))
    analyze_casg_statistics(root_path)