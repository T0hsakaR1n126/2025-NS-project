import pandas as pd
import networkx as nx
import requests
import comtradeapicall
import wbgapi
import importlib.metadata
version = importlib.metadata.version("comtradeapicall")
print("comtradeapicall version:", version)
# comtrade api subscription key (from comtradedeveloper.un.org)
subscription_key = 'e24dfdc3c644425086e31dd6392bca1b'
directory = '<OUTPUT DIR>'  # output directory for downloaded files
proxy_url = '<PROXY URL>'  # optional if you need proxy url
exportdataframe=pd.DataFrame()
for year in range(2000,2021):
    exportdataframe = pd.concat([exportdataframe,(comtradeapicall.getFinalData(typeCode='C', freqCode='A', clCode='HS', period=str(year),
                                        reporterCode=None, cmdCode='TOTAL', flowCode='X', partnerCode=None,
                                        partner2Code=None,
                                        customsCode=None, motCode=None, maxRecords=None, format_output='JSON',
                                        aggregateBy=None, breakdownMode='classic', countOnly=None, includeDesc=True, subscription_key=subscription_key))],ignore_index=True)
exportdataframe
importdataframe=pd.DataFrame()
for year in range(2000,2021):
    importdataframe = pd.concat([importdataframe,(comtradeapicall.getFinalData(typeCode='C', freqCode='A', clCode='HS', period=str(year),
                                        reporterCode=None, cmdCode='TOTAL', flowCode='M', partnerCode=None,
                                        partner2Code=None,
                                        customsCode=None, motCode=None, maxRecords=None, format_output='JSON',
                                        aggregateBy=None, breakdownMode='classic', countOnly=None, includeDesc=True, subscription_key=subscription_key))],ignore_index=True)
importdataframe
indicator = 'NY.GDP.MKTP.KD'   # GDP (constant 2015 US$)

gdpdataframe = wbgapi.data.DataFrame(
    indicator,
    economy='all',
    time=range(2000, 2021),
)

gdpdataframe = gdpdataframe.reset_index()
gdpdataframe = gdpdataframe.rename(columns={
    'economy': 'iso3c',
    'time': 'year',
    indicator: 'gdp_const_2015_usd'
})
gdpdataframe
