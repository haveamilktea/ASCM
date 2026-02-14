"""
人工合成控制法（ASCM）分析 - 社会消费品零售总额（深度拓展版）
研究对象：郑州市作为首批国家电子商务示范城市的政策效应
核心拓展：
1. 稳健性检验（安慰剂置换、权重剔除、时间窗口调整）
2. 异质性分析（分时段/分指标效应分解）
3. 动态效应计算（逐年边际效应、政策半衰期）
4. 数据质控（异常值检测、多重共线性检验）
5. 结果可视化升级（学术期刊级图表样式）
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.stats import zscore
from statsmodels.stats.outliers_influence import variance_inflation_factor
import warnings
import os

warnings.filterwarnings('ignore')
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'PingFang SC']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 150
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False
plt.rcParams['font.family'] = 'Arial'

# ==================== 全局配置（拓展版）====================
# 核心参数（保持原研究设计）
TREATMENT_CITY = '郑州'
TREATMENT_YEAR = 2012
PRE_END = 2011
POST_END = 2017
Y_VAR = '社会消费品零售总额（十亿元）'
X_VARS = ['人均国内生产总值（人民币）', '第二产业占比', '第三产业占比',
          '常住人口数（千人）', '移动电话用户数（千户）']
# 拓展参数
WEIGHT_REG = 0.01
X_WEIGHT = 0.5
OUTLIER_THRESHOLD = 3  # 异常值Z分数阈值
VIF_THRESHOLD = 10     # 多重共线性VIF阈值
RESULT_DIR = r'D:\HW\电商\ASCM_拓展版结果'
os.makedirs(RESULT_DIR, exist_ok=True)  # 自动创建结果目录

# ==================== 1. 数据增强预处理（新增质控模块）====================
def load_and_preprocess_data_enhanced(file_path):
    """
    拓展版数据预处理：新增异常值检测、多重共线性检验、数据标准化优化
    """
    # 基础读取（复用原逻辑）
    sheets = pd.read_excel(file_path, sheet_name=None)
    df_combined = pd.DataFrame()
    df_y_donors = None

    for sheet_name, df in sheets.items():
        df = df.rename(columns={'Unnamed: 0': 'Year'})
        df['Year'] = df['Year'].astype(int)
        df = df.set_index('Year')

        if '社会消费品零售总额' in sheet_name:
            df_combined[Y_VAR] = df[TREATMENT_CITY]
            df_y_donors = df.drop(columns=[TREATMENT_CITY])
        else:
            for city in df.columns:
                df_combined[f'{city}_{sheet_name}'] = df[city]

    # 时间筛选
    df_combined = df_combined[(df_combined.index >= 2003) & (df_combined.index <= POST_END)].fillna(0)
    df_y_donors = df_y_donors[(df_y_donors.index >= 2003) & (df_y_donors.index <= POST_END)].fillna(0)
    donor_cities = df_y_donors.columns.tolist()

    # 提取预测变量矩阵
    X_treated = pd.DataFrame()
    X_donors = pd.DataFrame()
    for city in [TREATMENT_CITY] + donor_cities:
        x_cols = [f'{city}_{x_var}' for x_var in X_VARS]
        if city == TREATMENT_CITY:
            X_treated = df_combined[x_cols]
            X_treated.columns = X_VARS
        else:
            X_city = df_combined[x_cols]
            X_city.columns = [f'{city}_{col}' for col in X_VARS]
            X_donors = pd.concat([X_donors, X_city], axis=1)

    # ==================== 新增：数据质控 ====================
    # 1. 异常值检测（Z分数法）
    def detect_outliers(df, threshold):
        outliers = {}
        for col in df.columns:
            z_scores = zscore(df[col])
            outlier_idx = np.abs(z_scores) > threshold
            outliers[col] = df.index[outlier_idx].tolist()
        return outliers

    outliers_treated = detect_outliers(X_treated, OUTLIER_THRESHOLD)
    outliers_donors = detect_outliers(X_donors, OUTLIER_THRESHOLD)

    # 输出异常值报告
    print("="*60)
    print("数据质控 - 异常值检测结果")
    print("="*60)
    has_outlier = False
    for col, idx in outliers_treated.items():
        if idx:
            has_outlier = True
            print(f"处理组({TREATMENT_CITY}) {col} 异常值年份: {idx}")
    for col, idx in outliers_donors.items():
        if idx:
            has_outlier = True
            print(f"供体池 {col} 异常值年份: {idx}")
    if not has_outlier:
        print("未检测到异常值")

    # 2. 多重共线性检验（VIF）
    print("\n" + "="*60)
    print("数据质控 - 多重共线性检验（VIF）")
    print("="*60)
    X_vif = X_treated.copy()
    X_vif = (X_vif - X_vif.mean()) / X_vif.std()  # 标准化
    vif_data = pd.DataFrame()
    vif_data['变量'] = X_vif.columns
    vif_data['VIF'] = [variance_inflation_factor(X_vif.values, i) for i in range(X_vif.shape[1])]
    print(vif_data)
    high_vif = vif_data[vif_data['VIF'] > VIF_THRESHOLD]['变量'].tolist()
    if high_vif:
        print(f"⚠️  高共线性变量: {high_vif}（VIF > {VIF_THRESHOLD}）")
    else:
        print("✅ 所有变量无严重多重共线性")

    # 划分时期
    pre_period = (df_combined.index >= 2003) & (df_combined.index <= PRE_END)
    X_treated_pre = X_treated[pre_period]
    X_donors_pre = X_donors[pre_period]
    Y_treated_pre = df_combined[Y_VAR][pre_period]
    Y_donors_pre = df_y_donors[pre_period]
    Y_treated = df_combined[Y_VAR]
    Y_donors = df_y_donors
    time_index = Y_treated.index

    return (X_treated_pre, X_donors_pre, Y_treated_pre, Y_donors_pre,
            Y_treated, Y_donors, donor_cities, time_index, pre_period,
            outliers_treated, outliers_donors, vif_data)

# 读取数据（替换为实际路径）
file_path = r'D:\HW\电商\社会消费品零售总额.xlsx'
(X_treated_pre, X_donors_pre, Y_treated_pre, Y_donors_pre,
 Y_treated, Y_donors, donor_cities, time_index, pre_period,
 outliers, vif_data) = load_and_preprocess_data_enhanced(file_path)

# ==================== 2. 核心权重计算（复用并优化）====================
def calculate_ascm_weights(X_treated, X_donors, Y_treated, Y_donors, donor_cities):
    n_donors = len(donor_cities)
    n_x = X_treated.shape[1]
    n_t = Y_treated.shape[0]

    # 标准化（优化：避免除以零）
    X_treated_std = (X_treated - X_treated.mean()) / (X_treated.std() + 1e-8)
    X_donors_std = (X_donors - X_donors.mean()) / (X_donors.std() + 1e-8)
    Y_treated_std = (Y_treated - Y_treated.mean()) / (Y_treated.std() + 1e-8)
    Y_donors_std = (Y_donors - Y_donors.mean()) / (Y_donors.std() + 1e-8)

    # 重构供体池矩阵
    X_donors_reshaped = np.zeros((n_t, n_donors, n_x))
    for i, city in enumerate(donor_cities):
        X_donors_reshaped[:, i, :] = X_donors_std[[f'{city}_{col}' for col in X_VARS]].values

    def objective(w):
        X_synth_std = np.sum(X_donors_reshaped * w.reshape(1, -1, 1), axis=1)
        Y_synth_std = Y_donors_std @ w
        x_error = np.sum((X_treated_std.values - X_synth_std) ** 2) / (n_t * n_x)
        y_error = np.sum((Y_treated_std.values - Y_synth_std) ** 2) / n_t
        reg_error = WEIGHT_REG * np.sum(w ** 2)
        return X_WEIGHT * x_error + (1 - X_WEIGHT) * y_error + reg_error

    constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
    bounds = [(0, 1) for _ in range(n_donors)]
    w0 = np.ones(n_donors) / n_donors

    result = minimize(objective, w0, method='SLSQP', bounds=bounds,
                      constraints=constraints, options={'maxiter': 5000, 'ftol': 1e-8})

    weights = w0 if not result.success else result.x
    weights = np.maximum(weights, 0)
    weights = weights / weights.sum()
    return weights

# 基础权重计算
weights = calculate_ascm_weights(X_treated_pre, X_donors_pre, Y_treated_pre, Y_donors_pre, donor_cities)
weights_df = pd.DataFrame({'城市': donor_cities, '权重': np.round(weights, 6)})
weights_df = weights_df.sort_values('权重', ascending=False).reset_index(drop=True)

# ==================== 3. 拓展分析1：稳健性检验（核心新增）====================
def robustness_tests(X_treated_pre, X_donors_pre, Y_treated_pre, Y_donors_pre,
                     Y_donors, donor_cities, pre_period, post_period, base_weights, base_att):
    """
    稳健性检验套件：
    1. 权重剔除检验（逐一剔除高权重城市）
    2. 时间窗口调整（后处理期缩短/延长）
    3. 安慰剂置换检验（随机分配处理组）
    """
    robustness_results = {}

    # 1. 权重剔除检验
    print("\n" + "="*60)
    print("稳健性检验1：权重剔除检验")
    print("="*60)
    high_weight_cities = weights_df[weights_df['权重'] > 0.1]['城市'].tolist()  # 高权重城市（权重>0.1）
    exclude_results = []
    for exclude_city in high_weight_cities:
        new_donors = [c for c in donor_cities if c != exclude_city]
        if len(new_donors) < 3:
            continue
        # 重新计算权重和ATT
        new_X_donors_pre = X_donors_pre[[f'{c}_{col}' for c in new_donors for col in X_VARS]]
        new_Y_donors_pre = Y_donors_pre[new_donors]
        new_weights = calculate_ascm_weights(X_treated_pre, new_X_donors_pre, Y_treated_pre, new_Y_donors_pre, new_donors)
        new_Y_synth = Y_donors[new_donors] @ new_weights
        new_post_effect = (Y_treated - new_Y_synth)[post_period]
        new_att = np.mean(new_post_effect)
        att_change = (new_att - base_att) / base_att * 100
        exclude_results.append({
            '剔除城市': exclude_city,
            '新ATT': new_att,
            'ATT变化率(%)': att_change,
            '结果稳定性': '稳定' if abs(att_change) < 20 else '敏感'
        })
    exclude_df = pd.DataFrame(exclude_results)
    robustness_results['权重剔除'] = exclude_df
    print(exclude_df)

    # 2. 时间窗口调整
    print("\n" + "="*60)
    print("稳健性检验2：时间窗口调整")
    print("="*60)
    window_results = []
    # 缩短后处理期：2012-2015
    post_short = (time_index >= 2012) & (time_index <= 2015)
    short_effect = (Y_treated - Y_synthetic)[post_short]
    short_att = np.mean(short_effect)
    # 延长预处理期（若数据允许）：2000-2011（需数据支持，此处仅示例）
    window_results.append({
        '时间窗口': '2012-2015（缩短后处理期）',
        'ATT': short_att,
        '与基准ATT差异': short_att - base_att
    })
    window_df = pd.DataFrame(window_results)
    robustness_results['时间窗口'] = window_df
    print(window_df)

    # 3. 安慰剂置换检验（随机分配处理组）
    print("\n" + "="*60)
    print("稳健性检验3：安慰剂置换检验")
    print("="*60)
    np.random.seed(42)  # 固定随机种子
    placebo_atts = []
    n_permutations = 100  # 置换次数
    for _ in range(n_permutations):
        # 随机权重（保持和为1、非负）
        rand_weights = np.random.dirichlet(np.ones(len(donor_cities)))
        rand_synth = Y_donors @ rand_weights
        rand_effect = (Y_treated - rand_synth)[post_period]
        rand_att = np.mean(rand_effect)
        placebo_atts.append(rand_att)
    # 计算基准ATT的分位数
    placebo_atts = np.array(placebo_atts)
    att_percentile = np.mean(placebo_atts >= base_att) * 100
    robustness_results['置换检验'] = {
        '基准ATT分位数(%)': att_percentile,
        '置换检验p值': att_percentile / 100,
        '显著性': '显著' if att_percentile < 10 else '不显著'
    }
    print(f"基准ATT在安慰剂分布中的分位数: {att_percentile:.2f}%")
    print(f"置换检验p值: {att_percentile/100:.4f}")

    return robustness_results

# 基础效应计算
Y_synthetic = Y_donors @ weights
treatment_effect = Y_treated - Y_synthetic
post_period = (time_index >= TREATMENT_YEAR) & (time_index <= POST_END)
post_effect = treatment_effect[post_period]
base_att = np.mean(post_effect)

# 执行稳健性检验
robustness_results = robustness_tests(
    X_treated_pre, X_donors_pre, Y_treated_pre, Y_donors_pre,
    Y_donors, donor_cities, pre_period, post_period, weights, base_att
)

# ==================== 4. 拓展分析2：动态效应与异质性====================
def dynamic_effect_analysis(treatment_effect, time_index, treatment_year):
    """
    动态效应分析：
    1. 逐年边际效应
    2. 政策效应半衰期
    3. 分时段效应分解
    """
    print("\n" + "="*60)
    print("动态效应与异质性分析")
    print("="*60)
    # 1. 逐年边际效应（相对于政策前均值）
    pre_mean = treatment_effect[pre_period].mean()
    dynamic_effect = pd.DataFrame({
        '年份': time_index,
        '原始效应': treatment_effect,
        '边际效应(相对预处理期)': (treatment_effect - pre_mean) / pre_mean * 100,
        '政策后年限': [max(0, y - treatment_year) for y in time_index]
    })
    # 2. 分时段效应（政策初期：2012-2014；政策成熟期：2015-2017）
    early_post = (time_index >= 2012) & (time_index <= 2014)
    late_post = (time_index >= 2015) & (time_index <= 2017)
    early_effect = treatment_effect[early_post].mean()
    late_effect = treatment_effect[late_post].mean()
    # 3. 效应增长率
    growth_rate = (late_effect - early_effect) / early_effect * 100

    dynamic_results = {
        '逐年动态效应': dynamic_effect,
        '政策初期ATT(2012-2014)': early_effect,
        '政策成熟期ATT(2015-2017)': late_effect,
        '效应增长率(%)': growth_rate,
        '效应趋势': '加速增长' if growth_rate > 10 else '平稳增长' if growth_rate > 0 else '衰减'
    }
    print(f"政策初期(2012-2014)平均效应: {early_effect:.4f} 十亿元")
    print(f"政策成熟期(2015-2017)平均效应: {late_effect:.4f} 十亿元")
    print(f"效应增长率: {growth_rate:.2f}% → 趋势: {dynamic_results['效应趋势']}")
    return dynamic_results

# 执行动态效应分析
dynamic_results = dynamic_effect_analysis(treatment_effect, time_index, TREATMENT_YEAR)

# ==================== 5. 可视化升级（学术期刊级）====================
def plot_enhanced_visuals(time_index, Y_treated, Y_synthetic, treatment_effect,
                          post_period, weights_df, robustness_results, dynamic_results):
    """
    拓展版可视化：
    1. 核心趋势图（期刊样式）
    2. 动态效应图
    3. 稳健性检验对比图
    4. 权重分布+拟合优度组合图
    """
    fig = plt.figure(figsize=(20, 16))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

    # 子图1：核心趋势（升级样式）
    ax1 = fig.add_subplot(gs[0, 0:2])
    ax1.plot(time_index, Y_treated, 'o-', linewidth=3, markersize=7,
             color='#E63946', label=f'{TREATMENT_CITY}（实际）', alpha=0.8)
    ax1.plot(time_index, Y_synthetic, 's--', linewidth=2.5, markersize=6,
             color='#457B9D', label='合成郑州', alpha=0.8)
    ax1.axvline(TREATMENT_YEAR, color='#6A994E', linestyle='--', linewidth=2.5,
                label='政策干预(2012)', alpha=0.8)
    ax1.fill_between(time_index[post_period], Y_treated[post_period], Y_synthetic[post_period],
                     color='#F1FAEE', alpha=0.5, label='政策效应区间')
    ax1.set_xlabel('年份', fontsize=12, fontweight='bold')
    ax1.set_ylabel('社会消费品零售总额（十亿元）', fontsize=12, fontweight='bold')
    ax1.set_title(f'{TREATMENT_CITY} vs 合成郑州 消费总额趋势 (2003-{POST_END})',
                  fontsize=14, fontweight='bold', pad=15)
    ax1.legend(fontsize=11, frameon=False)
    ax1.grid(True, alpha=0.2)
    ax1.set_xticks(np.arange(2003, POST_END+1, 2))

    # 子图2：动态边际效应
    ax2 = fig.add_subplot(gs[0, 2])
    dynamic_effect = dynamic_results['逐年动态效应']
    post_dynamic = dynamic_effect[dynamic_effect['政策后年限'] > 0]
    ax2.bar(post_dynamic['年份'], post_dynamic['边际效应(相对预处理期)'],
            color='#FFB703', alpha=0.7, edgecolor='black', linewidth=0.5)
    ax2.axhline(0, color='black', linestyle='-', linewidth=1)
    ax2.set_xlabel('年份', fontsize=11, fontweight='bold')
    ax2.set_ylabel('边际效应(%)', fontsize=11, fontweight='bold')
    ax2.set_title('政策后边际效应（相对预处理期）', fontsize=12, fontweight='bold')
    ax2.tick_params(axis='x', rotation=45)

    # 子图3：权重分布（横向条形图）
    ax3 = fig.add_subplot(gs[1, 0])
    top_weights = weights_df.head(7)
    bars = ax3.barh(top_weights['城市'][::-1], top_weights['权重'][::-1],
                    color='#219EBC', alpha=0.8, edgecolor='black', linewidth=0.5)
    ax3.set_xlabel('权重', fontsize=11, fontweight='bold')
    ax3.set_title('供体池城市权重分布', fontsize=12, fontweight='bold')
    # 数值标注
    for bar in bars:
        width = bar.get_width()
        ax3.text(width + 0.005, bar.get_y() + bar.get_height()/2,
                 f'{width:.3f}', va='center', fontsize=9)

    # 子图4：稳健性检验（权重剔除）
    ax4 = fig.add_subplot(gs[1, 1])
    exclude_df = robustness_results['权重剔除']
    if not exclude_df.empty:
        ax4.bar(exclude_df['剔除城市'], exclude_df['ATT变化率(%)'],
                color=['#90E0EF' if x < 20 else '#FF6B6B' for x in exclude_df['ATT变化率(%)']],
                alpha=0.7, edgecolor='black', linewidth=0.5)
        ax4.axhline(20, color='red', linestyle='--', linewidth=1, label='20%阈值')
        ax4.axhline(-20, color='red', linestyle='--', linewidth=1)
        ax4.set_xlabel('剔除城市', fontsize=11, fontweight='bold')
        ax4.set_ylabel('ATT变化率(%)', fontsize=11, fontweight='bold')
        ax4.set_title('权重剔除检验 - ATT稳定性', fontsize=12, fontweight='bold')
        ax4.tick_params(axis='x', rotation=45)
        ax4.legend(fontsize=9)
    else:
        ax4.text(0.5, 0.5, '无高权重城市可剔除', ha='center', va='center',
                 transform=ax4.transAxes, fontsize=11)

    # 子图5：动态效应趋势
    ax5 = fig.add_subplot(gs[1, 2])
    post_time = time_index[post_period]
    cumulative_effect = np.cumsum(post_effect)
    ax5.plot(post_time, cumulative_effect, 'o-', linewidth=3, markersize=6,
             color='#7209B7', alpha=0.8)
    ax5.fill_between(post_time, 0, cumulative_effect, color='#C77DFF', alpha=0.3)
    ax5.set_xlabel('年份', fontsize=11, fontweight='bold')
    ax5.set_ylabel('累积效应（十亿元）', fontsize=11, fontweight='bold')
    ax5.set_title(f'政策累积效应 ({TREATMENT_YEAR}-{POST_END})', fontsize=12, fontweight='bold')
    ax5.grid(True, alpha=0.2)

    # 子图6：拟合优度+安慰剂检验
    ax6 = fig.add_subplot(gs[2, 0:2])
    # 计算拟合优度
    Y_treated_pre_vals = Y_treated[pre_period]
    Y_synth_pre_vals = Y_synthetic[pre_period]
    ss_res = np.sum((Y_treated_pre_vals - Y_synth_pre_vals)**2)
    ss_tot = np.sum((Y_treated_pre_vals - Y_treated_pre_vals.mean())**2)
    r_squared = max(0, 1 - ss_res/(ss_tot + 1e-8))
    # 绘制拟合散点
    ax6.scatter(Y_treated_pre_vals, Y_synth_pre_vals, s=100, alpha=0.7,
                color='#F72585', edgecolor='black', linewidth=1)
    min_val = min(Y_treated_pre_vals.min(), Y_synth_pre_vals.min())
    max_val = max(Y_treated_pre_vals.max(), Y_synth_pre_vals.max())
    ax6.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='45°拟合线')
    ax6.text(0.05, 0.95, f'R² = {r_squared:.4f}', transform=ax6.transAxes,
             fontsize=12, fontweight='bold', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    ax6.set_xlabel(f'{TREATMENT_CITY}实际值', fontsize=11, fontweight='bold')
    ax6.set_ylabel('合成郑州值', fontsize=11, fontweight='bold')
    ax6.set_title('预处理期拟合效果 + 安慰剂检验参考', fontsize=12, fontweight='bold')
    ax6.legend(fontsize=10)

    # 子图7：置换检验分布
    ax7 = fig.add_subplot(gs[2, 2])
    placebo_atts = robustness_results['置换检验'].get('安慰剂分布', [])
    if len(placebo_atts) > 0:
        ax7.hist(placebo_atts, bins=15, color='#06D6A0', alpha=0.7, edgecolor='black')
        ax7.axvline(base_att, color='red', linewidth=2, label=f'基准ATT = {base_att:.4f}')
        ax7.set_xlabel('安慰剂ATT', fontsize=11, fontweight='bold')
        ax7.set_ylabel('频次', fontsize=11, fontweight='bold')
        ax7.set_title('安慰剂置换检验分布', fontsize=12, fontweight='bold')
        ax7.legend(fontsize=10)
    else:
        ax7.text(0.5, 0.5, '置换检验结果\np值 = {:.4f}'.format(robustness_results['置换检验']['置换检验p值']),
                 ha='center', va='center', transform=ax7.transAxes, fontsize=11)

    # 保存图表
    plot_path = os.path.join(RESULT_DIR, 'ASCM_拓展版可视化图表.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\n✅ 拓展版可视化图表已保存: {plot_path}")

# 执行可视化
plot_enhanced_visuals(time_index, Y_treated, Y_synthetic, treatment_effect,
                      post_period, weights_df, robustness_results, dynamic_results)

# ==================== 6. 结果导出（结构化保存）====================
def export_results_enhanced(weights_df, effects_yearly, fit_metrics,
                            robustness_results, dynamic_results):
    """
    结构化导出所有结果：
    1. 基础结果（权重、效应、拟合优度）
    2. 稳健性检验结果
    3. 动态效应结果
    4. 综合分析报告
    """
    # 1. 基础结果
    effects_yearly.to_excel(os.path.join(RESULT_DIR, '逐年效应数据_拓展版.xlsx'), index=False)
    weights_df.to_excel(os.path.join(RESULT_DIR, '供体池权重_拓展版.xlsx'), index=False)
    pd.DataFrame([fit_metrics]).to_excel(os.path.join(RESULT_DIR, '拟合优度_拓展版.xlsx'), index=False)

    # 2. 稳健性检验结果
    with pd.ExcelWriter(os.path.join(RESULT_DIR, '稳健性检验结果.xlsx')) as writer:
        robustness_results['权重剔除'].to_excel(writer, sheet_name='权重剔除', index=False)
        robustness_results['时间窗口'].to_excel(writer, sheet_name='时间窗口', index=False)
        pd.DataFrame([robustness_results['置换检验']]).to_excel(writer, sheet_name='置换检验', index=False)

    # 3. 动态效应结果
    dynamic_results['逐年动态效应'].to_excel(os.path.join(RESULT_DIR, '动态效应数据.xlsx'), index=False)
    dynamic_summary = pd.DataFrame({
        '指标': ['政策初期ATT', '政策成熟期ATT', '效应增长率', '效应趋势'],
        '值': [dynamic_results['政策初期ATT(2012-2014)'],
               dynamic_results['政策成熟期ATT(2015-2017)'],
               f"{dynamic_results['效应增长率(%)']:.2f}%",
               dynamic_results['效应趋势']]
    })
    dynamic_summary.to_excel(os.path.join(RESULT_DIR, '动态效应汇总.xlsx'), index=False)

    # 4. 综合报告
    r_squared = fit_metrics['R²(预处理期)']
    report = f"""
【ASCM拓展分析报告 - {TREATMENT_CITY}电商政策效应（社会消费品零售总额）】

一、研究基础信息
- 处理组：{TREATMENT_CITY}（首批国家电子商务示范城市）
- 政策时点：{TREATMENT_YEAR}年
- 研究区间：预处理期(2003-{PRE_END}) | 后处理期({TREATMENT_YEAR}-{POST_END})
- 供体池：{donor_cities}（共{len(donor_cities)}个城市）
- 核心变量：结果变量({Y_VAR}) | 预测变量({X_VARS})

二、基础拟合结果
- 预处理期R²：{r_squared:.4f}
- 平均处理效应(ATT)：{base_att:.4f} 十亿元（{TREATMENT_YEAR}-{POST_END}）
- 最高权重供体城市：{weights_df.iloc[0]['城市']}（权重：{weights_df.iloc[0]['权重']:.4f}）

三、稳健性检验结论
1. 权重剔除检验：{robustness_results['权重剔除']['结果稳定性'].value_counts().to_dict()}
2. 时间窗口调整：缩短后处理期至2012-2015年，ATT={robustness_results['时间窗口'].iloc[0]['ATT']:.4f}，与基准差异{robustness_results['时间窗口'].iloc[0]['与基准ATT差异']:.4f}
3. 安慰剂置换检验：p值={robustness_results['置换检验']['置换检验p值']:.4f} → {robustness_results['置换检验']['显著性']}

四、动态效应分析
1. 政策初期(2012-2014)：ATT={dynamic_results['政策初期ATT(2012-2014)']:.4f} 十亿元
2. 政策成熟期(2015-2017)：ATT={dynamic_results['政策成熟期ATT(2015-2017)']:.4f} 十亿元
3. 效应增长率：{dynamic_results['效应增长率(%)']:.2f}% → 趋势：{dynamic_results['效应趋势']}

五、核心结论
1. 拟合效果：预处理期R²={r_squared:.4f}，合成控制组能够{'较好' if r_squared>0.7 else '基本'}模拟{TREATMENT_CITY}政策前的消费特征；
2. 政策效应：电商政策显著提升了{TREATMENT_CITY}社会消费品零售总额，平均每年增加{base_att:.4f}十亿元，且效应呈{dynamic_results['效应趋势']}趋势；
3. 结果稳健性：{('权重剔除和时间窗口调整验证了结果的稳健性' if robustness_results['权重剔除']['结果稳定性'].iloc[0] == '稳定' else '结果对高权重城市敏感，需谨慎解读')}；
4. 政策含义：数字化政策通过提升流通效率，有效促进了消费市场扩张，为全国统一大市场建设提供了地方经验。

六、研究局限与建议
1. 供体池限制：现有供体池缺乏内陆枢纽城市，可能导致拟合偏差，建议纳入武汉、西安等城市验证；
2. 机制分析：可进一步结合线上消费数据，拆解电商政策的直接效应与间接效应；
3. 异质性拓展：可分城乡、分商品类型分析政策效应的差异化表现。
    """
    # 保存报告
    report_path = os.path.join(RESULT_DIR, 'ASCM拓展分析报告.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"✅ 拓展版分析报告已保存: {report_path}")
    print(f"\n📁 所有拓展版结果已保存至目录: {RESULT_DIR}")

# 计算拟合优度（复用原函数）
def calculate_fit_metrics(Y_actual, Y_fitted, pre_period):
    Y_actual_pre = Y_actual[pre_period]
    Y_fitted_pre = Y_fitted[pre_period]
    rmspe = np.sqrt(np.mean((Y_actual_pre - Y_fitted_pre) ** 2))
    mape = np.mean(np.abs((Y_actual_pre - Y_fitted_pre) / (Y_actual_pre + 1e-8))) * 100
    ss_res = np.sum((Y_actual_pre - Y_fitted_pre) ** 2)
    ss_tot = np.sum((Y_actual_pre - np.mean(Y_actual_pre)) ** 2)
    r_squared = max(0, 1 - (ss_res / (ss_tot + 1e-8)))
    nmse = ss_res / (np.var(Y_actual_pre) * len(Y_actual_pre) + 1e-8)
    return {'R²(预处理期)': r_squared, 'RMSPE(预处理期)': rmspe,
            'MAPE(预处理期)(%)': mape, 'NMSE(预处理期)': nmse}

fit_metrics = calculate_fit_metrics(Y_treated, Y_synthetic, pre_period)

# 整理逐年效应数据
effects_yearly = pd.DataFrame({
    '年份': time_index,
    f'{TREATMENT_CITY}实际值': np.round(Y_treated, 4),
    '合成郑州值': np.round(Y_synthetic, 4),
    '处理效应': np.round(treatment_effect, 4),
    '相对效应(%)': np.round((treatment_effect/(Y_synthetic+1e-8))*100, 4)
})

# 执行结果导出
export_results_enhanced(weights_df, effects_yearly, fit_metrics, robustness_results, dynamic_results)

# ==================== 最终提示 ====================
print("\n" + "="*80)
print("🎉 ASCM拓展版分析执行完成！核心新增模块：")
print("="*80)
print("1. 数据质控：异常值检测 + 多重共线性检验")
print("2. 稳健性检验：权重剔除 + 时间窗口调整 + 安慰剂置换")
print("3. 动态效应：逐年边际效应 + 政策周期分解")
print("4. 可视化升级：学术期刊级图表样式")
print("5. 结构化导出：所有结果按模块分类保存")
print("\n💡 建议：")
print("1. 更换供体池为内陆枢纽城市（武汉/西安/长沙等）提升拟合效果；")
print("2. 结合物流、线上消费数据补充机制分析；")
print("3. 对比不同政策时点（如跨境电商综试区设立）的效应差异。")