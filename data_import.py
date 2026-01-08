import time
import pandas as pd
import os
import logging
from tqdm import tqdm
import comtradeapicall
import wbgapi

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class FinalDataDownloader:
    def __init__(self, subscription_key: str, data_dir: str = './data'):
        self.subscription_key = subscription_key
        self.data_dir = data_dir
        self.raw_dir = os.path.join(data_dir, 'raw')
        
        os.makedirs(self.raw_dir, exist_ok=True)
        
        print(f"Data downloader initialized")
        print(f"Data will be saved to: {self.raw_dir}")
    
    def download_trade_data(self):
        print("Downloading trade data from UN Comtrade (2000-2020)...")
        
        start_year = 2000
        end_year = 2020
        
        export_frames = []
        for year in tqdm(range(start_year, end_year + 1), desc="Export Data"):
            try:
                df = comtradeapicall.getFinalData(
                    typeCode='C', freqCode='A', clCode='HS', 
                    period=str(year),
                    reporterCode=None, cmdCode='TOTAL', flowCode='X', 
                    partnerCode=None, partner2Code=None,
                    customsCode=None, motCode=None, maxRecords=None, 
                    format_output='JSON',
                    aggregateBy=None, breakdownMode='classic', 
                    countOnly=None, includeDesc=True, 
                    subscription_key=self.subscription_key
                )
                
                if not df.empty:
                    df['year'] = year
                    export_frames.append(df)
                else:
                    logger.warning(f"Empty export data for {year}")
                    
            except Exception as e:
                logger.error(f"Failed to download export data for {year}: {e}")
            
            time.sleep(0.5)
        
        if export_frames:
            export_data = pd.concat(export_frames, ignore_index=True)
            print(f"Export data downloaded: {len(export_data)} records")
            
            export_file = os.path.join(self.raw_dir, 'comtrade_export_raw.csv')
            export_data.to_csv(export_file, index=False)
            print(f"Export data saved: {export_file}")
        else:
            logger.error("No export data downloaded")
            return False
        
        import_frames = []
        for year in tqdm(range(start_year, end_year + 1), desc="Import Data"):
            try:
                df = comtradeapicall.getFinalData(
                    typeCode='C', freqCode='A', clCode='HS', 
                    period=str(year),
                    reporterCode=None, cmdCode='TOTAL', flowCode='M', 
                    partnerCode=None, partner2Code=None,
                    customsCode=None, motCode=None, maxRecords=None, 
                    format_output='JSON',
                    aggregateBy=None, breakdownMode='classic', 
                    countOnly=None, includeDesc=True, 
                    subscription_key=self.subscription_key
                )
                
                if not df.empty:
                    df['year'] = year
                    import_frames.append(df)
                else:
                    logger.warning(f"Empty import data for {year}")
                    
            except Exception as e:
                logger.error(f"Failed to download import data for {year}: {e}")
            
            time.sleep(0.5)
        
        if import_frames:
            import_data = pd.concat(import_frames, ignore_index=True)
            print(f"Import data downloaded: {len(import_data)} records")
            
            import_file = os.path.join(self.raw_dir, 'comtrade_import_raw.csv')
            import_data.to_csv(import_file, index=False)
            print(f"Import data saved: {import_file}")
            return True
        else:
            logger.error("No import data downloaded")
            return False
    
    def download_gdp_data(self):
        print("Downloading GDP data from World Bank (2000-2020)...")
        
        try:
            indicator = 'NY.GDP.MKTP.KD'
    
            gdp_df = wbgapi.data.DataFrame(
                indicator,
                economy='all',
                time=range(2000, 2021),
            )
            
            formatted_gdp = self._convert_gdp_format(gdp_df)
            
            gdp_file = os.path.join(self.raw_dir, 'world_bank_gdp.csv')
            formatted_gdp.to_csv(gdp_file, index=False)
            print(f"GDP data saved: {gdp_file}")
            
            return formatted_gdp
            
        except Exception as e:
            logger.error(f"Failed to download GDP data: {e}")
            raise
    
    def _convert_gdp_format(self, gdp_df):       
        gdp_df = gdp_df.reset_index()
        
        iso3_column = None
        year_columns = []
        
        for col in gdp_df.columns:
            col_str = str(col).strip().upper()
            
            if col_str in ['ISO3C', 'ECONOMY', 'COUNTRY']:
                iso3_column = col
            elif ('YR' in col_str or 'YEAR' in col_str):
                year_columns.append(col)
        
        if iso3_column is None:
            iso3_column = gdp_df.columns[0]
        
        if not year_columns:
            year_columns = [col for col in gdp_df.columns if col != iso3_column]
        
        long_format_data = []
        
        for idx, row in gdp_df.iterrows():
            iso3 = row[iso3_column]
            
            if pd.isna(iso3) or str(iso3).strip() in ['', 'WLD']:
                continue
            
            for col in year_columns:
                gdp_value = row[col]
                
                col_str = str(col).strip().upper()
                year_match = str(col_str)[-4:]
                
                if year_match.isdigit() and len(year_match) == 4:
                    year = int(year_match)
                else:
                    continue
                
                try:
                    if pd.isna(gdp_value):
                        continue
                    
                    gdp_numeric = float(gdp_value)
                    
                    if gdp_numeric > 0:
                        long_format_data.append({
                            'iso3c': str(iso3).strip().upper(),
                            'year': year,
                            'gdp_const_2015_usd': gdp_numeric
                        })
                except (ValueError, TypeError):
                    continue
        
        formatted_df = pd.DataFrame(long_format_data)
        
        if formatted_df.empty:
            logger.error("Empty data after format conversion")
            return formatted_df
        
        formatted_df = self._clean_gdp_data(formatted_df)
        
        print(f"Records: {len(formatted_df)} records")
        print(f"Year range: {formatted_df['year'].min()} - {formatted_df['year'].max()}")
        print(f"Number of countries: {formatted_df['iso3c'].nunique()}")
        
        return formatted_df
    
    def _clean_gdp_data(self, df):
        original_len = len(df)
        
        df['iso3c'] = df['iso3c'].astype(str).str.strip().str.upper()
        df['year'] = pd.to_numeric(df['year'], errors='coerce')
        df['gdp_const_2015_usd'] = pd.to_numeric(df['gdp_const_2015_usd'], errors='coerce')
        
        df = df.dropna(subset=['iso3c', 'year', 'gdp_const_2015_usd'])
        df = df[df['gdp_const_2015_usd'] > 0]
        
        invalid_codes = ['WLD', '0', '000', 'NAN', 'NULL']
        df = df[~df['iso3c'].isin(invalid_codes)]
        
        df = df[(df['year'] >= 2000) & (df['year'] <= 2020)]
        
        df['year'] = df['year'].astype(int)
        
        removed_count = original_len - len(df)
        if removed_count > 0:
            print(f"Removed {removed_count} invalid rows")
        
        return df
    
    def run_complete_download(self):
        try:
            print("\n" + "=" * 70)
            print("STEP 1: Download World Bank GDP Data")
            print("=" * 70)
            
            try:
                gdp_data = self.download_gdp_data()
                
                if gdp_data is not None and not gdp_data.empty:
                    print(f"GDP data downloaded successfully!")
                else:
                    print("Failed to download GDP data")
            except Exception as e:
                print(f"Failed to download GDP data: {e}")
                return False
            
            print("\n" + "=" * 70)
            print("STEP 2: Download UN Comtrade Trade Data")
            print("=" * 70)
            
            success = self.download_trade_data()
            
            if success:
                print(f"\nTrade data downloaded!")
            else:
                print("✗ Failed to download trade data")
            
            print("\n" + "=" * 70)
            print("STEP 3: Download Summary")
            print("=" * 70)
            
            print("Generated files:")
            if os.path.exists(self.raw_dir):
                files = sorted(os.listdir(self.raw_dir))
                for file in files:
                    file_path = os.path.join(self.raw_dir, file)
                    size_mb = os.path.getsize(file_path) / (1024 * 1024)
                    print(f"  {file} ({size_mb:.2f} MB)")
            
            print("\nAll data downloaded!")
            
            return True
            
        except KeyboardInterrupt:
            print("\n\n Download interrupted by user")
            return False
        except Exception as e:
            print(f"\n Error during download: {e}")
            return False

def main():
    subscription_key = 'd6c7bcc720cb4aa280e0f4957737f014'
    
    downloader = FinalDataDownloader(
        subscription_key=subscription_key,
        data_dir='./data'
    )
    
    downloader.run_complete_download()

if __name__ == "__main__":
    main()