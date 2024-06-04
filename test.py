import joblib
import pandas as pd
import numpy as np

# 加载保存的模型
loaded_model = joblib.load('数据/model.joblib')

# 创建模拟数据
# 请根据你的特征列和数据类型进行调整
simulated_data = {
    '性别': [1],  # 1表示男性
    '年龄': [40],
    '高密度脂蛋白胆固醇': [1.28],
    '低密度脂蛋白胆固醇': [3.31],
    '极低密度脂蛋白胆固醇': [1.27],
    '甘油三酯': [2.87],
    '总胆固醇': [5.86],
    '脉搏': [70],
    '舒张压': [69],
    '高血压史': [1],  # 0表示无高血压史
    '尿素氮': [5.0],
    '尿酸': [243.3],
    '肌酐': [50],
    '体重检查结果': [3],  # 2表示偏重
}

simulated_df = pd.DataFrame(simulated_data)

# 使用加载的模型进行预测
predictions = loaded_model.predict(simulated_df)

# 输出预测结果
print("预测结果:", predictions)