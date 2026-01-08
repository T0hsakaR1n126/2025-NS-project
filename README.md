# NS-project Guide

1. Data Source
  (1). Bilateral Trade Data
  Source: UN Comtrade Database. If you want to use api to download the data, please first use the URL: https://comtradeplus.un.org/BilateralData to register a free account. After login, please first click "EXPLORE APIS" and then click "Free APIs" to generate a free subscription key. You can use your key by replacing it in line 294 in data_import.py.
  (2). Real GDP Data
  Source: World Bank Databank; URL: https://databank.worldbank.org/source/world-development-indicators

2. Compile And Run the Code
  (1). Package to Install:
  run ```pip install pandas numpy matplotlib scipy networkx tqdm comtradeapicall wbgapi``` to install related package.
  (2). Compile And Run the Code
    (a) run ```python data_import.py```.
    (b) run ```python nw_build.py```.
    (c) run ```python nw_analysis.py```.

Note: If you use the API of UN Comtrade Database too often, the api will denied your access about 19 hours. After 19 hours, you can use it again.