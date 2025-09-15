import xgboost as xgb
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

# 1. 生成虚构的信贷数据（1000个用户）
np.random.seed(42)  # 固定随机数，保证结果可复现

# 特征说明：
# 收入（月收入，单位：千元）
# 负债比（月负债/月收入，0-1之间）
# 征信查询次数（近3个月）
# 历史逾期次数（近1年）
# 工作年限（年）
data = {
    'income': np.random.normal(15, 5, 1000).clip(3, 50),  # 收入：3-50千，均值15
    'debt_ratio': np.random.uniform(0.1, 0.8, 1000),      # 负债比：10%-80%
    'credit_inquiries': np.random.randint(0, 10, 1000),   # 查询次数：0-9次
    'overdue_times': np.random.randint(0, 6, 1000),       # 逾期次数：0-5次
    'work_years': np.random.randint(0, 20, 1000)          # 工作年限：0-19年
}

# 构造标签：是否违约（1=违约，0=正常）
# 逻辑：收入低、负债高、查询多、逾期多的人更容易违约（模拟真实规律）
debt_risk = (
    (data['income'] < 8) * 0.3 +          # 低收入增加风险
    (data['debt_ratio'] > 0.6) * 0.2 +    # 高负债增加风险
    (data['credit_inquiries'] > 5) * 0.2 +# 频繁查询增加风险
    (data['overdue_times'] > 2) * 0.3     # 多次逾期增加风险
)
# 基于风险值生成标签（概率化）
data['default'] = (debt_risk + np.random.normal(0, 0.1, 1000) > 0.5).astype(int)

# 转成DataFrame，方便查看
df = pd.DataFrame(data)
print("前5条数据：")
print(df.head())

# 2. 划分特征(X)和标签(y)
X = df.drop('default', axis=1)  # 所有特征
y = df['default']               # 标签（是否违约）

# 3. 拆分训练集和测试集（7:3）
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# 4. 转换为XGBoost专用格式
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

# 5. 设置模型参数（二分类任务，预测是否违约）
params = {
    'objective': 'binary:logistic',  # 二分类，输出概率
    'eval_metric': 'logloss',        # 评估指标：对数损失
    'max_depth': 3,                  # 树深度（控制复杂度）
    'learning_rate': 0.1,            # 学习率
    'verbosity': 0                   # 不输出训练日志
}

# 6. 训练模型
model = xgb.train(params, dtrain, num_boost_round=100)

# 7. 预测测试集
# 输出违约概率（0-1之间），大于0.5视为会违约
y_pred_proba = model.predict(dtest)
y_pred = (y_pred_proba > 0.5).astype(int)

# 8. 评估模型效果
accuracy = accuracy_score(y_test, y_pred)
print(f"\n模型准确率：{accuracy:.2f}")  # 通常在75%-85%之间

# 混淆矩阵：展示预测结果细节
print("\n混淆矩阵（行=实际值，列=预测值）：")
# 矩阵含义：[ [真正常，假违约], [假正常，真违约] ]
print(confusion_matrix(y_test, y_pred))
