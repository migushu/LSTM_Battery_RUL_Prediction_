'''
 !/usr/bin/env python
 -*- coding: utf-8 -*-
 @Time    : 2019/7/12 16:26
 @Author  : Tang
 @File    : read.py
 @Software: PyCharm
 @reference:
 @description: read .xls file .get wanted sheets and output them with csv format.
'''

import xlrd
import pandas as pd

'''
---------------sheets: filename1------------------
Global_Info
Channel_1-043_1
Channel_1-043_2
Statistics_1-043 ---->one cell's cycle data
Channel_1-046_1
Channel_1-046_2
Statistics_1-046 ---->one cell's cycle data
---------------sheets: filename2------------------
Global_Info
Channel_1-005
Statistics_1-005
Channel_1-013
Statistics_1-013
Channel_1-017
Statistics_1-017
Channel_1-019
Statistics_1-019
Channel_1-022
Statistics_1-022
Channel_1-023
Statistics_1-023
'''

filename1 = "SKS-IP0418-4_15_4_2-CYCLE-180629"  # linux需把反斜杠改成斜杠
filename2 = "SK-180801"
sheet_names1 = ['Statistics_1-043','Statistics_1-046']
sheet_names2 = ['Statistics_1-005','Statistics_1-013','Statistics_1-017','Statistics_1-019','Statistics_1-022','Statistics_1-023']
file_path = "data2/"

def xls_2_csv_pd(filename,sheets_name):
    for sheet_name in sheets_name:
        sheet = pd.read_excel(file_path+filename+'.xls', sheet_name=sheet_name)
        sheet.to_csv(file_path+sheet_name + '.csv', encoding='utf-8', index=False)


def xls_sheets_name(filename):
    book = xlrd.open_workbook(filename)
    for sheet in book.sheets():
        print(sheet.name)

if __name__ == '__main__':
    # xls_sheets_name()
    xls_2_csv_pd(filename1,sheet_names1)
    xls_2_csv_pd(filename2,sheet_names2)