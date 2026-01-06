# test_gdp_filter.py
import pandas as pd
import numpy as np
from network_builder import WTWDataImporter

def test_gdp_filter():
    """测试GDP数据过滤"""
    print("测试GDP数据过滤")
    print("="*60)
    
    # 初始化导入器
    importer = WTWDataImporter(
        subscription_key='dummy',
        data_dir='./data'
    )
    
    # 加载GDP数据
    print("加载GDP数据...")
    gdp_data = importer.load_gdp_data()
    
    # 测试不同年份
    test_years = [2000, 2005, 2010, 2015, 2020]
    
    for year in test_years:
        print(f"\n测试{year}年:")
        print(f"  GDP数据形状: {gdp_data.shape}")
        print(f"  GDP数据年份数据类型: {gdp_data['year'].dtype}")
        
        # 方法1：直接过滤
        result1 = gdp_data[gdp_data['year'] == year]
        print(f"  直接过滤: {len(result1)}条记录")
        
        if len(result1) > 0:
            print(f"  样本数据:")
            print(result1.head(3).to_string())
        
        # 方法2：使用.query()
        try:
            result2 = gdp_data.query(f"year == {year}")
            print(f"  使用.query(): {len(result2)}条记录")
        except:
            print(f"  使用.query()失败")
        
        # 方法3：使用.loc
        try:
            result3 = gdp_data.loc[gdp_data['year'] == year]
            print(f"  使用.loc: {len(result3)}条记录")
        except:
            print(f"  使用.loc失败")

if __name__ == "__main__":
    test_gdp_filter()