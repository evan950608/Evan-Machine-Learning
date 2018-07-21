import pandas as pd
import numpy as np
import openpyxl
import pyodbc

import bs4,os

import myLibs.myHtml as myHtml
import myLibs.myHtml_selenium as myHtml_selenium
import myLibs.myCache as myCache
import myLibs.myFile as myFile
import myLibs.myList as myList

class ModuleExcel_Format():
    '''
    fmt = xls_writer.book.add_format() # '{'num_format': '0.0%'} #,##0.00', '0.0%','h:mm:ss AM/PM'
    worksheet.set_column('D:D', None, format1)      '''
    def __init__(self, column_str, xls_format_dict):
        self.column_str = column_str
        self.xls_format_dict = xls_format_dict
        

def preparing_trained_data(f,sep):
    df = pd.read_csv(f, sep) #, index_col=0
    
    ###### edit data ######
    #for idx,row in df.iterrows(): df.at[idx, 'cn'] =  1 if row['cn'] >= 1 else 0 
    #df['cn'] = df.apply(lambda r: 1 if r['cn']>=1 else 0 ,axis=1) 
    df.loc[df.cn > 1, ['cn']] = 1
    df['ip_nonif'] = df.apply(lambda r: 'ip' if ',{0},'.format(r.MATTER_DEPT) in ',P,JP,T,H2,K2,'  else 'nonip' ,axis=1)         
    
    dept_not = ['A1','A2','A3','A4','TL','GA','X2']
    
    ##### query/filter #####
    df = df.query(' MATTER_DEPT not in @dept_not ')    
    df = df.query(" PPRACTICE_CODE not in ['999'] ")        
    df = df[ (df.PRACTICE_CODE.str.len() >= 5) & (df.PRACTICE_CODE.str.contains('999')==False)]     
    return df

def pd_groupby_data(df, gp_name, apply_lambda, new_index_name):
    gp = df.groupby(gp_name).apply(apply_lambda).reset_index(name=new_index_name) 
    '''
    gp = pandas_groupby_data(df, gp_name, lambda x: x.cn.count(), 'matter_count')
    #gp = df.groupby(gp_name).apply(lambda x: x.cn.count()).reset_index(name='matter_count')
    #gp2 = df.groupby(gp_name).apply(lambda x: x.cn.sum()).reset_index(name='file_count')
    '''    
    return gp

def pd_to_excel(f, sheet_name, df, list_module_excel_Format):
        writer = pd.ExcelWriter(f, engine = 'openpyxl')
        if myFile.is_file(f):
            book = openpyxl.load_workbook(f)        
            writer.book = book        
        df.to_excel(writer, sheet_name=sheet_name, index=False)        
        
        #### formatting #####
        sheet = writer.sheets[sheet_name]
        #sheet['D3'].number_format ='0.0%'
        for item in list_module_excel_Format:
            for cell in sheet[item.column_str]:
                for key, value in item.xls_format_dict.items():
                    if key == 'number_format':  cell.number_format = value
        #fmt = xls_writer.book.add_format(item.xls_format_dict) # '{'num_format': '0.0%'} #,##0.00', '0.0%','h:mm:ss AM/PM''
        #worksheet.set_column(, None, fmt)  
        
        writer.save()
        print(f)
        
def doing_group_analsis(df, gp_name, f_xls):
    df_gp = pd_groupby_data(df, gp_name, lambda x: x.cn.count(), 'matter_count')
    gp = pd_groupby_data(df, gp_name, lambda x: x.cn.sum(), 'file_count')        
    df_gp = pd.merge(df_gp, gp, on=gp_name)        
    df_gp.eval(' file_percent =  (file_count / matter_count) ' , inplace=True) #gp['file_percent'] =   np.round( gp['file_count'] / gp['matter_count'] ,decimals=2)
    
    list_module_excel_Format = [ ModuleExcel_Format('D:D', {'number_format': '0.0%'}) ]
    pd_to_excel(f_xls, gp_name, df_gp, list_module_excel_Format)

def get_df_from_odbc(conn_str, sql, f_csv=None):
    conn = pyodbc.connect(conn_str)
    df = pd.read_sql(sql, conn, index_col=None)
    conn.close()
    if f_csv != None:
        df.to_csv(f_csv, index=False)
    return df
    

def doing_basic_df():
    conn_str = 'DRIVER={SQL Server};' + 'SERVER={0};DATABASE={1}; Trusted connection=YES'.format('db-matter','lee_matter')
    
    ##### pg ####
    f_csv = os.path.join(myFile.get_download_path(r'out\csv') , 'pg.csv')
    sql = "select * from PRACTICE_GROUP where  C_NAME <> '(空白)' and PPRACTICE_CODE not like 'stf%'  and PRACTICE_CODE not in ('000') and PRACTICE_CODE=PPRACTICE_CODE order by PPRACTICE_CODE "
    df = get_df_from_odbc(conn_str, sql, f_csv)

    ##### pa ####
    f_csv = os.path.join(myFile.get_download_path(r'out\csv') , 'pa.csv')
    sql = "select * from PRACTICE_GROUP where  C_NAME <> '(空白)' and PPRACTICE_CODE not like 'stf%'  and PRACTICE_CODE not in ('000') and PRACTICE_CODE<>PPRACTICE_CODE and pa_select='Y' order by PPRACTICE_CODE "
    df = get_df_from_odbc(conn_str, sql, f_csv)


def main():
    debug_def = ''
    if debug_def == '':
        f = os.path.join(myFile.get_download_path(r'out\csv') , 'data.csv')
        f_test = os.path.join(myFile.get_download_path(r'out\csv') , 'test.csv')
        f_xls = os.path.join( myFile.get_download_path(r'out\csv') , 'data.xlsx')   
        if myFile.is_file(f_xls): myFile.remove(f_xls)
        df = preparing_trained_data(f, sep='\t')

        #df = df.sort_index(by=['PRACTICE_CODE', 'MATTER_DEPT'], ascending=[True, False])
        #r = pd.pivot_table(df, values = 'cn', index=['PPRACTICE_CODE'], columns = 'MATTER_DEPT').reset_index() 
        #df.pivot_table(index="PAR NAME",values=["value"],aggfunc={'value':lambda x: x[df.iloc[x.index]['DESTCD']=='E'].sum()*100.0/x.sum()})

        r1 = df.pivot_table( values = 'cn', index=['PPRACTICE_CODE'], columns = 'MATTER_DEPT',aggfunc='count') # fill_value=0
        r2 = df.pivot_table( values = 'cn', index=['PPRACTICE_CODE'], columns = 'MATTER_DEPT',aggfunc='sum') # fill_value=0
        #r = r2 / r1
        r = df.pivot_table( values = 'cn', index=['PPRACTICE_CODE'], columns = 'MATTER_DEPT',aggfunc={'cn': lambda x: x.sum()/x.count()}) # fill_value=0
        #r.to_csv(f_test)
        print(r)
        return
        gp_name = 'ip_nonif'
        doing_group_analsis(df, gp_name, f_xls)
        

        gp_name = 'MATTER_DEPT'
        doing_group_analsis(df, gp_name, f_xls)
        
        gp_name = 'PPRACTICE_CODE'        
        doing_group_analsis(df, gp_name, f_xls)

        gp_name = 'PRACTICE_CODE'        
        doing_group_analsis(df, gp_name, f_xls)
        
        pass
    elif debug_def == '':
        pass
    pass

if __name__ == '__main__':
    main()
    pass

