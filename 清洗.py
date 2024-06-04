import pandas as pd

# 读取Excel文件
df = pd.read_excel('数据/dia.xls')

# 打印数据框的前几行，以确认是否正确读取
print(df.head())

# 删除重复行
df = df.drop_duplicates()

# 将性别列中的0替换为'女'，1替换为'男'
df['性别'] = df['性别'].replace({0: '女', 1: '男'})

# 将体重检查结果列中的值进行替换
df['体重检查结果'] = df['体重检查结果'].replace({0: '较瘦', 1: '正常', 2: '偏重', 3: '肥胖'})

# 将高血压史列中的值进行替换
df['高血压史'] = df['高血压史'].replace({0: '无', 1: '有'})

# 将是否糖尿病列中的值进行替换
df['是否糖尿病'] = df['是否糖尿病'].replace({0: '否', 1: '是'})

# 计算每列的缺失值数量
missing_values = df.isnull().sum()

# 打印每列的缺失值数量
print("每列缺失值数量:")
print(missing_values)

df.to_csv('数据/数据.csv',index=False)