import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, \
    confusion_matrix
import matplotlib.pyplot as plt
import joblib

# 读取数据
df = pd.read_csv('数据/数据.csv')
df = df.drop(['卡号'], axis=1)
df['性别'] = df['性别'].replace({'女': 0, '男': 1})
df['体重检查结果'] = df['体重检查结果'].replace({'较瘦': 0, '正常': 1, '偏重': 2, '肥胖': 3})
df['高血压史'] = df['高血压史'].replace({'无': 0, '有': 1})
df['是否糖尿病'] = df['是否糖尿病'].replace({'否': 0, '是': 1})

df['性别'] = df['性别'].astype(int)
df['年龄'] = df['年龄'].astype(int)
df['高血压史'] = df['高血压史'].astype(int)
df['是否糖尿病'] = df['是否糖尿病'].astype(int)

float_columns = ['高密度脂蛋白胆固醇', '低密度脂蛋白胆固醇', '极低密度脂蛋白胆固醇', '甘油三酯', '总胆固醇', '脉搏', '舒张压', '尿素氮', '尿酸', '肌酐']
df[float_columns] = df[float_columns].astype(float)

# 准备数据
X = df.drop(['是否糖尿病'], axis=1)
y = df['是否糖尿病']

# 划分数据集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 初始化和训练单一预测模型
dt_model = DecisionTreeClassifier(random_state=42)
svm_model = SVC(probability=True, random_state=42)

dt_model.fit(X_train, y_train)
svm_model.fit(X_train, y_train)

# 初始化和训练集成学习模型
ensemble_model = VotingClassifier(estimators=[('Decision Tree', dt_model), ('SVM', svm_model)], voting='soft')
ensemble_model.fit(X_train, y_train)

# 保存集成学习模型
joblib.dump(ensemble_model, '数据/ensemble_model.joblib')


# 评估单一预测模型
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_proba)
    confusion_mat = confusion_matrix(y_test, y_pred)

    fpr, tpr, _ = roc_curve(y_test, y_proba)

    return accuracy, precision, recall, f1, roc_auc, confusion_mat, fpr, tpr


dt_accuracy, dt_precision, dt_recall, dt_f1, dt_roc_auc, dt_confusion_mat, dt_fpr, dt_tpr = evaluate_model(dt_model,
                                                                                                           X_test,
                                                                                                           y_test)
svm_accuracy, svm_precision, svm_recall, svm_f1, svm_roc_auc, svm_confusion_mat, svm_fpr, svm_tpr = evaluate_model(
    svm_model, X_test, y_test)
ensemble_accuracy, ensemble_precision, ensemble_recall, ensemble_f1, ensemble_roc_auc, ensemble_confusion_mat, ensemble_fpr, ensemble_tpr = evaluate_model(
    ensemble_model, X_test, y_test)

# 打印评估结果
print("Decision Tree 模型评估结果:")
print("准确率:", dt_accuracy)
print("精确度:", dt_precision)
print("召回率:", dt_recall)
print("F1 值:", dt_f1)
print("ROC AUC:", dt_roc_auc)
print("混淆矩阵:\n", dt_confusion_mat)

print("\nSVM 模型评估结果:")
print("准确率:", svm_accuracy)
print("精确度:", svm_precision)
print("召回率:", svm_recall)
print("F1 值:", svm_f1)
print("ROC AUC:", svm_roc_auc)
print("混淆矩阵:\n", svm_confusion_mat)

print("\n集成学习模型评估结果:")
print("准确率:", ensemble_accuracy)
print("精确度:", ensemble_precision)
print("召回率:", ensemble_recall)
print("F1 值:", ensemble_f1)
print("ROC AUC:", ensemble_roc_auc)
print("混淆矩阵:\n", ensemble_confusion_mat)

# 交叉验证评估集成学习模型
ensemble_cv_scores = cross_val_score(ensemble_model, X, y, cv=5, scoring='accuracy')
mean_cv_accuracy = ensemble_cv_scores.mean()
print("\n集成学习模型交叉验证平均准确率:", mean_cv_accuracy)

# 绘制 ROC 曲线
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']  # 设置中文显示
plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题

plt.figure(figsize=(10, 7))
plt.plot(dt_fpr, dt_tpr, label='决策树 (AUC = %0.2f)' % dt_roc_auc)
plt.plot(svm_fpr, svm_tpr, label='支持向量机 (AUC = %0.2f)' % svm_roc_auc)
plt.plot(ensemble_fpr, ensemble_tpr, label='集成学习模型 (AUC = %0.2f)' % ensemble_roc_auc)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('假正例率')
plt.ylabel('真正例率')
plt.title('ROC曲线')
plt.legend(loc="lower right")
plt.savefig('result.png')
plt.show()
'''
Decision Tree 模型评估结果:
准确率: 0.7524752475247525
精确度: 0.6951219512195121
召回率: 0.6951219512195121
F1 值: 0.6951219512195121
ROC AUC: 0.7433943089430893
混淆矩阵:
 [[95 25]
 [25 57]]

SVM 模型评估结果:
准确率: 0.698019801980198
精确度: 0.6019417475728155
召回率: 0.7560975609756098
F1 值: 0.6702702702702702
ROC AUC: 0.7963414634146342
混淆矩阵:
 [[79 41]
 [20 62]]

集成学习模型评估结果:
准确率: 0.7524752475247525
精确度: 0.6951219512195121
召回率: 0.6951219512195121
F1 值: 0.6951219512195121
ROC AUC: 0.8473577235772358
混淆矩阵:
 [[95 25]
 [25 57]]

集成学习模型交叉验证平均准确率: 0.7256686862716122

进程已结束,退出代码0

'''