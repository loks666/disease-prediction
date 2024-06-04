import json
from django.http import HttpResponse
from django.shortcuts import render
import sqlite3
import os
import pandas as pd
import joblib

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, 'sql.db')


def query(sql):
    connection = sqlite3.connect(DB_PATH)
    cursor = connection.cursor()
    cursor.execute(sql)
    result = cursor.fetchall()
    connection.close()
    return result


def insert(sql):
    connection = sqlite3.connect(DB_PATH)
    cursor = connection.cursor()
    cursor.execute(sql)
    connection.commit()
    connection.close()


def login(request):
    if request.method == 'GET':
        return render(request, 'login.html')
    else:
        param = json.loads(request.body.decode('utf-8'))
        username = param['username']
        password = param['password']
        sql = 'SELECT * FROM `tb_user` WHERE username = "{0}" AND password = "{1}" LIMIT 0,1'.format(username, password)
        res = query(sql)
        if not res:
            data = '该用户未注册，请注册后再登录'
            return HttpResponse(data)
        else:
            data = "登录成功"
            return HttpResponse(data)


def register(request):
    if request.method == 'GET':
        return render(request, 'register.html')
    else:
        param = json.loads(request.body.decode('utf-8'))
        username = param['username']
        password = param['password']
        sql = 'SELECT * FROM tb_user WHERE username = "{0}" AND password = "{1}" LIMIT 0,1'.format(username, password)
        res = query(sql)
        if res:
            data = '该用户已经注册'
            return HttpResponse(data)
        else:
            sql = 'INSERT INTO tb_user(`username`, `password`) VALUES ("{0}", "{1}")'.format(username, password)
            insert(sql)
            data = "注册成功"
            return HttpResponse(data)


def look_data(request):
    df = pd.read_csv('数据.csv')
    # 将 DataFrame 转换为列表
    data_list = df.values.tolist()
    data = []
    for i in data_list:
        a = {
            '性别': i[1],
            '年龄': i[2],
            '高密度脂蛋白胆固醇': i[3],
            '低密度脂蛋白胆固醇': i[4],
            '极低密度脂蛋白胆固醇': i[5],
            '甘油三酯': i[6],
            '总胆固醇': i[7],
            '脉搏': i[8],
            '舒张压': i[9],
            '高血压史': i[10],
            '尿素氮': i[11],
            '尿酸': i[12],
            '肌酐': i[13],
            '体重检查结果': i[14],
            '是否糖尿病': i[15]
        }
        data.append(a)
    data = {
        'data': data
    }
    return render(request, 'look_data.html', data)


def visualization(request):
    if request.method == 'GET':
        return render(request, 'visualization.html')
    else:
        result = []
        df = pd.read_csv('数据.csv')

        # 根据体重检查结果和是否糖尿病进行分组，并计算数量
        result_counts = df.groupby(['体重检查结果', '是否糖尿病']).size().unstack().fillna(0)
        weight_results = list(result_counts.index)
        is_diabetes = list(result_counts['是'])
        not_diabetes = list(result_counts['否'])
        result.append([weight_results, is_diabetes, not_diabetes])

        # 根据性别和是否糖尿病进行分组，并计算数量
        result_counts = df.groupby(['性别', '是否糖尿病']).size().unstack().fillna(0)
        # 将结果转换为ECharts饼图格式
        echarts_data_male = []
        echarts_data_female = []
        # 遍历结果，将数据转换为ECharts格式
        for gender in result_counts.index:
            diabetes_count = int(result_counts.loc[gender, '是']) if '是' in result_counts.columns else 0
            non_diabetes_count = int(result_counts.loc[gender, '否']) if '否' in result_counts.columns else 0

            if gender == '男':
                echarts_data_male.append({'name': '患病', 'value': diabetes_count})
                echarts_data_male.append({'name': '健康', 'value': non_diabetes_count})
            elif gender == '女':
                echarts_data_female.append({'name': '患病', 'value': diabetes_count})
                echarts_data_female.append({'name': '健康', 'value': non_diabetes_count})
        result.append([echarts_data_male, echarts_data_female])

        ageresult1 = df[df['是否糖尿病'] == '是'].groupby('年龄').size().reset_index(name='人数').sort_values(
            by='年龄').values.tolist()
        ageresult2 = df[df['是否糖尿病'] == '否'].groupby('年龄').size().reset_index(name='人数').sort_values(
            by='年龄').values.tolist()
        result.append([ageresult1, ageresult2])

        result_counts = df.groupby(['高血压史', '是否糖尿病']).size().unstack().fillna(0)
        # 提取结果中的列名和对应的值
        values_have_hypertension = result_counts.loc['有'].values.tolist()
        values_no_hypertension = result_counts.loc['无'].values.tolist()
        result.append([values_no_hypertension, values_have_hypertension])
        return HttpResponse(json.dumps(result))


def yc(request):
    if request.method == 'GET':
        return render(request, 'yc.html')
    else:
        a1 = int(request.POST.get('a1'))
        a2 = float(request.POST.get('a2'))
        a3 = float(request.POST.get('a3'))
        a4 = float(request.POST.get('a4'))
        a5 = float(request.POST.get('a5'))
        a6 = float(request.POST.get('a6'))
        a7 = float(request.POST.get('a7'))
        a8 = float(request.POST.get('a8'))
        a9 = float(request.POST.get('a9'))
        a10 = int(request.POST.get('a10'))
        a11 = float(request.POST.get('a11'))
        a12 = float(request.POST.get('a12'))
        a13 = float(request.POST.get('a13'))
        a14 = int(request.POST.get('a14'))

        # 加载保存的模型
        loaded_model = joblib.load('./static/ensemble_model.joblib')
        simulated_data = {
            '性别': [a1],
            '年龄': [a2],
            '高密度脂蛋白胆固醇': [a3],
            '低密度脂蛋白胆固醇': [a4],
            '极低密度脂蛋白胆固醇': [a5],
            '甘油三酯': [a6],
            '总胆固醇': [a7],
            '脉搏': [a8],
            '舒张压': [a9],
            '高血压史': [a10],
            '尿素氮': [a11],
            '尿酸': [a12],
            '肌酐': [a13],
            '体重检查结果': [a14],
        }

        simulated_df = pd.DataFrame(simulated_data)

        # 使用加载的模型进行预测
        predictions = loaded_model.predict(simulated_df)[0]
        if predictions == 1:
            return HttpResponse('！！请注意当前患病风险较高\n请保持良好生活习惯并及时就医')
        return HttpResponse('当前患病风险较低\n请继续保持良好生活习惯')

def echarts(request):
    if request.method == 'GET':
        return render(request, 'echarts.html')
    else:
        result = []
        df = pd.read_csv('数据.csv')

        # 根据体重检查结果和是否糖尿病进行分组，并计算数量
        result_counts = df.groupby(['体重检查结果', '是否糖尿病']).size().unstack().fillna(0)
        weight_results = list(result_counts.index)
        is_diabetes = list(result_counts['是'])
        not_diabetes = list(result_counts['否'])
        result.append([weight_results, is_diabetes, not_diabetes])

        # 根据性别和是否糖尿病进行分组，并计算数量
        result_counts = df.groupby(['性别', '是否糖尿病']).size().unstack().fillna(0)
        # 将结果转换为ECharts饼图格式
        echarts_data_male = []
        echarts_data_female = []
        # 遍历结果，将数据转换为ECharts格式
        for gender in result_counts.index:
            diabetes_count = int(result_counts.loc[gender, '是']) if '是' in result_counts.columns else 0
            non_diabetes_count = int(result_counts.loc[gender, '否']) if '否' in result_counts.columns else 0

            if gender == '男':
                echarts_data_male.append({'name': '患病', 'value': diabetes_count})
                echarts_data_male.append({'name': '健康', 'value': non_diabetes_count})
            elif gender == '女':
                echarts_data_female.append({'name': '患病', 'value': diabetes_count})
                echarts_data_female.append({'name': '健康', 'value': non_diabetes_count})
        result.append([echarts_data_male, echarts_data_female])

        ageresult1 = df[df['是否糖尿病'] == '是'].groupby('年龄').size().reset_index(name='人数').sort_values(
            by='年龄').values.tolist()
        ageresult2 = df[df['是否糖尿病'] == '否'].groupby('年龄').size().reset_index(name='人数').sort_values(
            by='年龄').values.tolist()
        result.append([ageresult1, ageresult2])

        result_counts = df.groupby(['高血压史', '是否糖尿病']).size().unstack().fillna(0)
        # 提取结果中的列名和对应的值
        values_have_hypertension = result_counts.loc['有'].values.tolist()
        values_no_hypertension = result_counts.loc['无'].values.tolist()
        result.append([values_no_hypertension, values_have_hypertension])
        return HttpResponse(json.dumps(result))
