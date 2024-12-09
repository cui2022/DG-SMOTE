import math
import random
import time
import operator
import pandas as pd
import numpy as np


# 保存处理过的全部pop
def store_excel(result, result_f, toolbox, pop, data, NUMS, SIZE, POP_SIZE, less_data):
    less_num = data.shape[1]
    coe_key = dict()
    for q in range(NUMS):
        coe = dict()
        for j in range(SIZE):
            values = pop[q * SIZE + j].fitness
            coe.update({q * SIZE + j: values})
        coe = dict(sorted(coe.items(), key=lambda x: x[1], reverse=True))

        judger = 0
        max__ = 0
        for key, value in coe.items():
            if value.values[0] != 0.0:
                if value.values[0] > max__:
                    max__ = value.values[0]
                individual = pop[key]
                func = toolbox.compile(expr=individual)
                pre_re = func(*data.T)
                judge = 0
                for kk in range(q):
                    if (result[kk, :] == pre_re).all():
                        judge = 1
                        break
                if judge == 0:
                    judger = 1
                    result_f = pd.concat([result_f, pd.DataFrame({'expression': [str(individual)]})])
                    result[q, :] = func(*data.T)
                    break
        if judger == 0:
            individual = pop[list(coe.keys())[0]]
            func = toolbox.compile(expr=individual)
            pre_re = func(*data.T)
            result_f = pd.concat([result_f, pd.DataFrame({'expression': [str(individual)]})])
            result[q, :] = pre_re
        coe_key.update(coe)
        print(coe)
    # 保存所有的生成的个体
    resultall = np.zeros(shape=(POP_SIZE, less_data.shape[0]))
    resultall_f = pd.DataFrame(columns=['expression'])
    for q, key in zip(range(POP_SIZE), coe_key.keys()):
        individual = pop[list(coe_key.keys())[q]]
        func = toolbox.compile(expr=individual)
        resultall_f = pd.concat([resultall_f, pd.DataFrame({'expression': [str(individual)]})])
        resultall[q, :] = func(*data.T)
    return resultall, resultall_f, result, result_f


def save_produce(data, i, file, NGEN):
    result_end = pd.DataFrame(data)
    file_name = '.\\GP_dataset\\' + file + '\\produce_set\\' + str(NGEN) + '\\' + file + " NEGN " + str(
        NGEN) + " seed " + str(
        i) + ".xlsx"
    writer = pd.ExcelWriter(file_name)  # 写入Excel文件
    result_end.to_excel(writer, 'less', header=None, index=None)  # ‘less’是写入excel的sheet名
    writer.save()


def save_expression(data1, data2, i, b_time, n_time, file, NGEN):
    data1 = pd.DataFrame(data1)
    data2.reset_index(drop=True, inplace=True)
    result_end = pd.concat([data1, data2], axis=1)
    result_end['TRAIN_TIME'] = n_time - b_time
    file_name = '.\\GP_dataset\\' + file + '\\expression_set\\' + str(
        NGEN) + '\\' + "data and expression " + file + " NEGN " + str(
        NGEN) + " seed " + str(
        i) + ".xlsx"
    writer = pd.ExcelWriter(file_name)
    result_end.to_excel(writer, 'less', header=None, index=None)
    writer.save()


def save_allproduce(data, i, file, NGEN):
    result_end = pd.DataFrame(data)
    file_name = '.\\GP_dataset\\' + file + '\\all_produce_set\\' + str(NGEN) + '\\' + file + " NEGN " + str(
        NGEN) + " seed " + str(
        i) + ".xlsx"
    writer = pd.ExcelWriter(file_name)  # 写入Excel文件
    result_end.to_excel(writer, 'less', header=None, index=None)  # ‘less’是写入excel的sheet名
    writer.save()


def save_allexpression(data1, data2, i, b_time, n_time, file, NGEN):
    data1 = pd.DataFrame(data1)
    data2.reset_index(drop=True, inplace=True)
    result_end = pd.concat([data1, data2], axis=1)
    result_end['TRAIN_TIME'] = n_time - b_time
    file_name = '.\\GP_dataset\\' + file + '\\all_expression_set\\' + str(
        NGEN) + '\\' + "data and expression " + file + " NEGN " + str(
        NGEN) + " seed " + str(
        i) + ".xlsx"
    writer = pd.ExcelWriter(file_name)  # 写入Excel文件
    result_end.to_excel(writer, 'less', header=None, index=None)  # ‘less’是写入excel的sheet名
    writer.save()