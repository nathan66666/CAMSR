# import pandas as pd
# import matplotlib.pyplot as plt
# import numpy as np

# # 设置学术风格
# plt.rcParams['font.family'] = 'serif'
# plt.rcParams['font.size'] = 11

# # 读取数据
# df = pd.read_csv('/data5/CAMSR/mamba_kmeans_013_4_4_train.csv')

# # 创建图形
# fig, ax = plt.subplots(figsize=(7, 5))

# # 绘制曲线
# ax.plot(df['Step'], df['Value'], 
#         marker='s',           # 方块标记
#         markersize=7,
#         linewidth=2,
#         color='#E63946',      # 红色
#         markerfacecolor='white',
#         markeredgewidth=1.5)

# # 设置坐标轴
# ax.set_xlabel('Training Steps', fontsize=12, fontweight='bold')
# ax.set_ylabel('PSNR (dB)', fontsize=12, fontweight='bold')
# ax.set_title('Training Performance', fontsize=13, fontweight='bold')

# # 设置刻度
# ax.set_xticks(df['Step'])
# ax.set_xticklabels([f'{s//1000}k' for s in df['Step']])  # 显示为 5k, 10k, 15k

# # 添加网格
# ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)

# # 添加数值标签（在点上方）
# for step, value in zip(df['Step'], df['Value']):
#     ax.annotate(f'{value:.2f}', 
#                 xy=(step, value),
#                 xytext=(0, 8),
#                 textcoords='offset points',
#                 ha='center',
#                 fontsize=10,
#                 fontweight='bold')

# # 计算并显示总提升
# improvement = df['Value'].iloc[-1] - df['Value'].iloc[0]
# ax.text(0.02, 0.95, f'+{improvement:.2f} dB', 
#         transform=ax.transAxes,
#         fontsize=11,
#         fontweight='bold',
#         color='green',
#         bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

# plt.tight_layout()
# plt.savefig('/data5/CAMSR/training_curve_enhanced.png', dpi=300, bbox_inches='tight')
# plt.savefig('/data5/CAMSR/training_curve_enhanced.pdf', bbox_inches='tight')  # 保存PDF用于论文
# plt.show()


import pandas as pd
import matplotlib.pyplot as plt

# 读取CSV文件
df = pd.read_csv('/data5/CAMSR/mamba_kmeans_013_4_4_train.csv')  # 替换为您的文件路径

# 查看数据
print(df)

# 创建图形
plt.figure(figsize=(8, 6))

# 绘制训练曲线
plt.plot(df['Step'], df['Value'], marker='o', linewidth=2, markersize=8)

# 设置标签和标题
plt.xlabel('Step', fontsize=12)
plt.ylabel('PSNR (dB)', fontsize=12)
plt.title('Training Curve', fontsize=14)

# 显示网格
plt.grid(True, alpha=0.3)

# 设置x轴刻度
plt.xticks(df['Step'])

# 在数据点上显示数值
for i, (step, value) in enumerate(zip(df['Step'], df['Value'])):
    plt.text(step, value + 0.05, f'{value:.2f}', ha='center', fontsize=10)

plt.tight_layout()
plt.savefig('/data5/CAMSR/training_curve.png', dpi=300)
plt.show()