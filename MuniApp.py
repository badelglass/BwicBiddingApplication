
import streamlit as st
import pandas as pd
import numpy as np
import io
from io import StringIO
from datetime import datetime

# -------------------------------------------------
# App config
# -------------------------------------------------
st.set_page_config(page_title="Sage Muni BWIC App With Algo Bids", layout="centered")
st.title("Sage Muni BWIC App With Algo Bids")

# -------------------------------------------------
# Helpers
# -------------------------------------------------
def classify_dur(dur_value: float):
    if pd.isna(dur_value): return 'Duration Not Available'
    if 0 <= dur_value < 1:   return '0-1'
    if 1 <= dur_value < 2:   return '1-2'
    if 2 <= dur_value < 3:   return '2-3'
    if 3 <= dur_value < 6:   return '3-6'
    if 6 <= dur_value < 10:  return '6-10'
    if dur_value >= 10:      return '10+'
    return 'Duration Not Available'

def ensure_str(series: pd.Series) -> pd.Series:
    return series.fillna('').astype(str).str.strip()

def normalize_flag(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.strip().str.lower()
    return s.replace({
        'y':'Y','yes':'Y','true':'Y','t':'Y','1':'Y',
        'n':'N','no':'N','false':'N','f':'N','0':'N'
    })

def get_first_existing_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    return None

# -------------------------------------------------
# App A — type-safe filter
# -------------------------------------------------
def filter_data_app_a(
    df: pd.DataFrame,
    include_ca: bool,
    include_ny: bool,
    include_prere: bool,
    start_year: int,
    end_year: int,
    coupon_lower: float,
    coupon_upper: float,
    selected_ratings: list[str],
    selected_durations: list[str],
    guarantors_vars: dict[str, bool],
) -> pd.DataFrame:

    # Dates: only require Mat Dt
    for dcol in ['Call Dt', 'Mat Dt', 'Put Dt']:
        if dcol in df.columns:
            df[dcol] = pd.to_datetime(df[dcol], format="%m/%d/%Y", errors='coerce')
    df = df.dropna(subset=['Mat Dt'])
    today = datetime.today()

    # Numerics
    for c in ['Mty Size','Issue Size','Min Denom','YrTSink','YrTMat','Cpn','Eff Dur','Par','BVAL Yld','ERP']:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')

    # Flags/text (upper for Y/N style; keep sector case)
    for c in ['AMT','144A','Taxable','Sinkable','Putable','Prere','Issue Type','St','Insured']:
        if c in df.columns:
            df[c] = normalize_flag(df[c])
    for c in ['BB Sector','Cpn Typ']:  # preserve case
        if c in df.columns:
            df[c] = df[c].astype(str).str.strip()

    # Exclusions
    if 'Mty Size' in df.columns:   df = df[df['Mty Size'] >= 2]
    if 'Issue Size' in df.columns: df = df[df['Issue Size'] >= 40]
    if 'Issue Type' in df.columns: df = df[~df['Issue Type'].str.upper().isin(['LIMITED','PRIVATE PLACEMENT'])]
    if 'Default' in df.columns:
        df['Default'] = df['Default'].astype(str).str.upper()
        df = df[df['Default'] != 'DEFAULT']
    if 'AMT' in df.columns:        df = df[df['AMT'] == 'N']
    if '144A' in df.columns:       df = df[df['144A'] == 'N']
    if 'Taxable' in df.columns:    df = df[df['Taxable'] == 'N']
    if 'Min Denom' in df.columns:  df = df[df['Min Denom'] <= 5000]
    if 'Par' in df.columns:        df = df[(df['Par'] == 500) | (df['Par'] >= 1000)]

    # Gas forwards: maturity replacement to align call window
    if {'BB Sector','Putable','Put Dt','Mat Dt'}.issubset(df.columns):
        df.loc[(df['BB Sector'] == 'Gas Forward Contract') & (df['Putable'] == 'Y'), 'Mat Dt'] = df.loc[
            (df['BB Sector'] == 'Gas Forward Contract') & (df['Putable'] == 'Y'), 'Put Dt'
        ]

    # Year & call window
    df['Call Wndw'] = (df['Mat Dt'].dt.year) - (df['Call Dt'].dt.year)
    df['Year'] = pd.to_numeric(df['Mat Dt'].dt.year, errors='coerce')
    df['Eff Dur'] = pd.to_numeric(df['Eff Dur'], errors='coerce')
    df = df.dropna(subset=['Eff Dur'])
    df['Call Wndw'] = pd.to_numeric(df['Call Wndw'], errors='coerce')
    mask_early = (df['Year'] <= 2039) & ((df['Call Wndw'] <= 5) | df['Call Wndw'].isna()) & (df['Eff Dur'] <= 10)
    mask_late  = (df['Year'] >= 2040) & (df['Call Wndw'] >= 7) & (df['Eff Dur'] >= 8)
    df = df[mask_early | mask_late]

    # Coupon type (allow Gas Forward)
    if {'Cpn Typ','BB Sector'}.issubset(df.columns):
        df = df[(df['Cpn Typ'] == 'FIXED') | (df['BB Sector'] == 'Gas Forward Contract')]

    # Sinkable rule
    df['YrTMat'] = pd.to_numeric(df['YrTMat'], errors='coerce')
    if {'Sinkable','YrTMat'}.issubset(df.columns):
        df = df[(df['Sinkable'] != 'Y') | ((df['Sinkable'] == 'Y') & (df['YrTMat'] > 20))]

    # Duration bucket
    df['Duration Category'] = df['Eff Dur'].apply(classify_dur)

    # Underwriter backfills for gas forwards
    sp_alias = get_first_existing_col(df, ['S&P','S&P'])
    if 'BB Sector' in df.columns and (df['BB Sector'] == 'Gas Forward Contract').any():
        for col in ['Mdy Undr','Fitch Undr','SP Undr','Moody','Fitch']:
            if col in df.columns:
                df[col] = df[col].astype(object)
        if 'Mdy Undr' in df.columns and 'Moody' in df.columns:
            df.loc[df['BB Sector'] == 'Gas Forward Contract','Mdy Undr'] = df.loc[df['BB Sector'] == 'Gas Forward Contract','Moody']
        if sp_alias and ('SP Undr' in df.columns):
            df.loc[df['BB Sector'] == 'Gas Forward Contract','SP Undr'] = df.loc[df['BB Sector'] == 'Gas Forward Contract', sp_alias]
        if 'Fitch Undr' in df.columns and 'Fitch' in df.columns:
            df.loc[df['BB Sector'] == 'Gas Forward Contract','Fitch Undr'] = df.loc[df['BB Sector'] == 'Gas Forward Contract','Fitch']
    
    def eliminate_na_mdy(date_str):
        try:
            return pd.to_datetime(date_str, format="%m/%d/%Y")
        except ValueError:
            return pd.NaT
    def eliminate_na_fitch(date_str):
        try:
            return pd.to_datetime(date_str, format="%m/%d/%Y")
        except ValueError:
            return pd.NaT
    def eliminate_na_sp(date_str):
        try:
            return pd.to_datetime(date_str, format="%m/%d/%Y")
        except ValueError:
            return pd.NaT
    
    df['Mdy Eff Dt'] = df['Mdy Eff Dt'].apply(eliminate_na_mdy)
    df['SP Eff Dt'] = df['SP Eff Dt'].apply(eliminate_na_fitch)
    df['Fitch Eff Dt'] = df['Fitch Eff Dt'].apply(eliminate_na_sp)

    df['Moody Days'] = (today - df['Mdy Eff Dt']).apply(lambda x: x.days if pd.notnull(x) else None)
    df['SP Days'] = (today - df['SP Eff Dt']).apply(lambda x: x.days if pd.notnull(x) else None)
    df['Fitch Days'] = (today - df['Fitch Eff Dt']).apply(lambda x: x.days if pd.notnull(x) else None)

    df['Days Since Rating Action'] = df[['Moody Days','SP Days','Fitch Days']].min(axis=1) #want to be alerted of any recent rating actions

    # Rating normalization
    replacement_dicts = {
        "Mdy Undr": {'Aaa':1,'MIG1':1,'Aaa/VMIG1':1,'Aa1':2,'Aa2':3,'Aa2/VMIG1':3,'Aa3':4,'A1':5,'A2':6,'MIG2':6,'A3':7,'Baa1':8,'Baa2':9,'Baa3':10,
                     'Ba1':11,'Ba2':12,'Ba3':13,'B1':14,'B2':15,'B3':16,'Caa1':17,'Caa2':18,'WD':99,'WR':99,'DNT N/A':99,'#N/A N/A':99,'NaN':99,None:99},
        "Fitch Undr": {'AAA':1,'AA+':2,'AA':3,'AA/F1+':3,'AA-':4,'A+':5,'A':6,'A-':7,'BBB+':8,'BBB':9,'BBB-':10,'BB+':11,'BB':12,'BB-':13,'B+':14,'B':15,'B-':16,'CCC+':17,'CCC':18,'WD':99,'D':99,'PIF':99,'DNT N/A':99,'#N/A N/A':99,'NaN':99,None:99},
        "SP Undr": {'AAA':1,'SP-1+':1,'AA+':2,'AA+/A-1+':2,'AA':3,'AA-':4,'A+':5,'A':6,'SP-2':6,'A-':7,'BBB+':8,'BBB':9,'BBB-':10,'BB+':11,'BB':12,'BB-':13,
                    'B+':14,'B':15,'B-':16,'CCC+':17,'CCC':18,'CC':19,'NR':99,'N.A.':99,'#N/A N/A':99,'DNT N/A':99,'NaN':99,None:99}
    }
    new_cols = {"Mdy Undr": 'Mdy_Undr_Score', "Fitch Undr": 'Fitch_Undr_Score', "SP Undr": 'SP_Undr_Score'}
    for col, repl in replacement_dicts.items():
        if col in df.columns:
            df[new_cols[col]] = df[col].map(lambda x: repl.get(x, 99))

    if all(c in df.columns for c in new_cols.values()):
        for c in new_cols.values():
            df[c] = pd.to_numeric(df[c], errors='coerce').fillna(99).astype(int)
        df['Highest Rating'] = df[list(new_cols.values())].min(axis=1)
        repl_band = {1:'AAA',2:'AA',3:'AA',4:'AA',5:'A',6:'A',7:'A',8:'BBB',9:'BBB',10:'BBB'}
        repl_band.update({i:'HY' for i in range(11,100)})
        df['Final Rating'] = df['Highest Rating'].replace(repl_band)

    # Region filters
    if ('St' in df.columns) and (not include_ca):
        df = df[df['St'] != 'CA']
    if ('St' in df.columns) and (not include_ny):
        df = df[df['St'] != 'NY']
    if ('Prere' in df.columns) and (not include_prere):
        df = df[df['Prere'] != 'Y']

    # Year & coupon
    df['Year'] = pd.to_numeric(df['Year'], errors='coerce')
    df['Cpn']  = pd.to_numeric(df['Cpn'], errors='coerce')
    df = df[df['Year'].between(int(start_year), int(end_year), inclusive='both')]
    df = df[df['Cpn'].between(float(coupon_lower), float(coupon_upper), inclusive='both')]

    # Selections
    if selected_ratings:
        df = df[df['Final Rating'].isin(selected_ratings)]
    if selected_durations:
        df = df[df['Duration Category'].isin(selected_durations)]

    # Guarantors
    if 'Guaranty Agmt' in df.columns:
        repl = {
            'ATHENE ANNUITY AND LIFE':'Athene','MERRILL LYNCH & CO INC':'BofA','BANK OF AMERICA CORP':'BofA',
            'BP PLC':'BP','BP CORP NORTH AMERICA INC':'BP','CITIGROUP INC':'Citi','CITIGROUP GLOBAL MKTS':'Citi',
            'DEUTSCHE BANK A.G.':'DB','GOLDMAN SACHS GROUP INC':'GS','GOLDMAN SACHS & COMPANY':'GS','JPMORGAN CHASE BANK':'JPM',
            'JP MORGAN CHASE & CO':'JPM','JP MORGAN CHASE BANK NA':'JPM','MORGAN STANLEY FINANCE':'MS','MORGAN STANLEY':'MS',
            'MORGAN STANLEY BANK NA':'MS','MORGAN STANLEY MUN FDG':'MS','NEW YORK LIFE INSURANCE C':'NYLIFE','ROYAL BANK OF CANADA':'RBC',
            'TD BANK N.A.':'TD'
        }
        df['Guaranty Agmt'] = df['Guaranty Agmt'].astype(str).str.strip().replace(repl)
        for value, var in guarantors_vars.items():
            if var:
                df = df[df['Guaranty Agmt'] != value]

    # Callable derivation
    if 'Callable' not in df.columns and 'Call Dt' in df.columns:
        df['Callable'] = df['Call Dt'].notna().map({True: 'Y', False: 'N'})

    return df

# -------------------------------------------------
# App B — Snowflake reference + G-Spread
# -------------------------------------------------
@st.cache_data(ttl=1000)
def load_reference_tables():
    conn = st.connection("snowflake")
    reference_df = conn.query("SELECT * FROM TRADER_SANDBOX.BRETT_SANDBOX.MUNI_ALADDIN_DATA", ttl=1000)
    curve_data   = conn.query("SELECT * FROM TRADER_SANDBOX.BRETT_SANDBOX.AAA_MUNI_CURVE", ttl=1000)
    return reference_df, curve_data

def prepare_reference_df(reference_df: pd.DataFrame) -> pd.DataFrame:
    reference_df = reference_df.copy()
    reference_df['Prepay Guarantor'] = ensure_str(reference_df.get('Prepay Guarantor', pd.Series(dtype=str))).str.upper()
    reference_df['GUARANTOR_NAME'] = reference_df.apply(
        lambda row: row['Prepay Guarantor']
        if (pd.isna(row.get('GUARANTOR_NAME')) or str(row.get('GUARANTOR_NAME')).strip() == '')
        else str(row.get('GUARANTOR_NAME')),
        axis=1
    )
    reference_df['GUARANTOR_NAME'] = ensure_str(reference_df['GUARANTOR_NAME']).str.upper()

    for dcol in ["MATURITY","NEXT_CALL_DATE","NEXT_SINKING_FUND_DATE","NEXT_PUT_DATE","Next Put Date Updated"]:
        if dcol in reference_df.columns:
            reference_df[dcol] = pd.to_datetime(reference_df[dcol], errors='coerce')

    reference_df['Sinkable'] = reference_df['NEXT_SINKING_FUND_DATE'].notna().map({True: 'Y', False: 'N'})
    if {'NEXT_PUT_DATE','Next Put Date Updated'}.issubset(reference_df.columns):
        reference_df['NEXT_PUT_DATE'] = reference_df.apply(
            lambda row: row['Next Put Date Updated']
            if (pd.isna(row['NEXT_PUT_DATE']) or row['NEXT_PUT_DATE'] == '') and pd.notna(row['Next Put Date Updated'])
            else row['NEXT_PUT_DATE'], axis=1
        )
    reference_df['Putable'] = reference_df['NEXT_PUT_DATE'].notna().map({True: 'Y', False: 'N'})
    reference_df.loc[(reference_df['Putable'] == 'Y'), 'MATURITY'] = reference_df.loc[(reference_df['Putable'] == 'Y'), 'NEXT_PUT_DATE']

    for col in ["COUPON","TIME_TO_MATURITY","YIELD_TO_WORST",'OAS','PRICE','DURATION','AMOUNT_ISSUED','YIELD','G_SPREAD']:
        if col in reference_df.columns:
            reference_df[col] = pd.to_numeric(reference_df[col], errors='coerce')

    reference_df['Maturity Year'] = reference_df['MATURITY'].dt.year
    current_year = datetime.now().year
    reference_df['Maturity Years From Today'] = (reference_df['Maturity Year'] - current_year).clip(lower=0)
    reference_df['Call Year']  = reference_df['NEXT_CALL_DATE'].dt.year
    reference_df['Call Wndw']  = reference_df.apply(lambda row: (row['Maturity Year'] - row['Call Year']) if pd.notna(row['Call Year']) else 0, axis=1)
    reference_df['Callable']   = reference_df['NEXT_CALL_DATE'].notna().map({True: 'Y', False: 'N'})
    reference_df['AMT']        = np.where(reference_df.get('AMT','N') == 'A', 'Y', 'N')

    rating_dict = {'Aaa':1,'AAA':1,'Aa1':2,'AA+':2,'Aa2':3,'AA':3,'Aa3':4,'AA-':4,'A1':5,'A+':5,'A2':6,'A':6,'A3':7,'A-':7,'Baa1':8,'BBB+':8,'BBB':9,'Baa2':9,'BBB-':10,'Baa3':10}
    if 'BARCLAYS_RATING' in reference_df.columns:
        reference_df['Final Rating'] = pd.to_numeric(reference_df['BARCLAYS_RATING'].replace(rating_dict), errors='coerce')

    if 'SAGE_BARCLAYS_MUNI_SECTOR_SAGE_BARC_MUNI_LEVEL_1' in reference_df.columns:
        reference_df['Muni Sector'] = reference_df['SAGE_BARCLAYS_MUNI_SECTOR_SAGE_BARC_MUNI_LEVEL_1'].replace({'Local':'Local GO','State':'State GO'})

    if 'DURATION' in reference_df.columns:
        reference_df['Duration Category'] = reference_df['DURATION'].apply(classify_dur)

    def assign_name(maturity_years, call_spread, callable_flag):
        if callable_flag == 'N': return "CallA"
        if maturity_years >= 28:
            if call_spread <= 23: return "CallA"
            elif 23 < call_spread <= 26: return "CallB"
            elif 26 < call_spread <= 30: return "CallC"
            else: return "CallD"
        if 24 < maturity_years < 28:
            if call_spread <= 17: return "CallA"
            elif 17 < call_spread <= 22: return 'CallB'
            elif 22 < call_spread <= 26: return 'CallC'
            else: return 'CallD'
        if 15 <= maturity_years <= 24:
            if call_spread <= 13: return "CallA"
            elif 13 < call_spread <= 18: return 'CallB'
            elif 18 < call_spread <= 22: return 'CallC'
            else: return 'CallD'
        if 10 < maturity_years < 15:
            if call_spread <= 4: return "CallA"
            elif 4 < call_spread <= 9: return 'CallB'
            elif 9 < call_spread <= 12: return 'CallC'
            else: return 'CallD'
        if maturity_years <= 10:
            if call_spread <= 3: return "CallA"
            elif 3 < call_spread <= 5: return 'CallB'
            else: return 'CallC'
    reference_df['Maturity-Call-Range'] = reference_df.apply(lambda r: assign_name(r.get('Maturity Years From Today',np.nan), r.get('Call Wndw',np.nan), r.get('Callable','N')), axis=1)

    for col in ['SP Undr','Sinkable','STATE','Muni Sector','Putable','Maturity-Call-Range','GUARANTOR_NAME']:
        if col in reference_df.columns:
            reference_df[col] = ensure_str(reference_df[col])

    return reference_df

def compute_gspread_display(input_df: pd.DataFrame,
                            reference_df: pd.DataFrame,
                            curve_data: pd.DataFrame,
                            metric_option: str,
                            adjustment_bp: int) -> pd.DataFrame:

    input_df = input_df.copy()

    # AAA curve yield merge by ceil(YrTMat)
    if 'YrTMat' in input_df.columns:
        input_df['YrTMat2'] = np.ceil(pd.to_numeric(input_df['YrTMat'], errors='coerce'))
    curve_data = curve_data.copy()
    curve_data['YrTMat2'] = curve_data['MATURITY']
    input_df = pd.merge(input_df, curve_data[['YrTMat2','YIELD']], on='YrTMat2', how='left')

    for col in ['Insured','SP Undr','Guaranty Agmt','Sinkable','Callable','Prere','St','Putable','BB Sector','BB Sector 3']:
        if col in input_df.columns:
            input_df[col] = ensure_str(input_df[col])
    sp_alias = get_first_existing_col(input_df, ['S&P','S&P'])
    if sp_alias:
        input_df[sp_alias] = ensure_str(input_df[sp_alias])
    if 'SP Undr' in input_df.columns and sp_alias and 'Insured' in input_df.columns:
        cond_missing_sp = (input_df['SP Undr'] == '') | (input_df['SP Undr'].str.lower().isin({'#n/a n/a'}))
        cond_uninsured  = (input_df['Insured'].str.upper() == 'N')
        input_df.loc[cond_uninsured & cond_missing_sp, 'SP Undr'] = input_df.loc[cond_uninsured & cond_missing_sp, sp_alias]

    input_df['Eff Dur'] = pd.to_numeric(input_df.get('Eff Dur', np.nan), errors='coerce')
    input_df = input_df.dropna(subset=['Eff Dur'])
    input_df['Duration Category'] = input_df['Eff Dur'].apply(classify_dur)
    input_df['COUPON'] = pd.to_numeric(input_df.get('Cpn', np.nan), errors='coerce')
    input_df['STATE']  = input_df.get('St', '')
    input_df['PREREFUNDED_FLAG'] = input_df.get('Prere', '')
    input_df['Call Wndw'] = pd.to_numeric(input_df.get('Call Wndw', 0), errors='coerce').fillna(0)

    # Callable bands
    def assign_name(maturity_years, call_spread, callable_flag):
        if callable_flag == 'N': return "CallA"
        if maturity_years >= 28:
            if call_spread <= 23: return "CallA"
            elif 23 < call_spread <= 26: return "CallB"
            elif 26 < call_spread <= 30: return "CallC"
            else: return "CallD"
        if 24 < maturity_years < 28:
            if call_spread <= 17: return "CallA"
            elif 17 < call_spread <= 22: return 'CallB'
            elif 22 < call_spread <= 26: return 'CallC'
            else: return 'CallD'
        if 15 <= maturity_years <= 24:
            if call_spread <= 13: return "CallA"
            elif 13 < call_spread <= 18: return 'CallB'
            elif 18 < call_spread <= 22: return 'CallC'
            else: return 'CallD'
        if 10 < maturity_years < 15:
            if call_spread <= 4: return "CallA"
            elif 4 < call_spread <= 9: return 'CallB'
            elif 9 < call_spread <= 12: return 'CallC'
            else: return 'CallD'
        if maturity_years <= 10:
            if call_spread <= 3: return "CallA"
            elif 3 < call_spread <= 5: return 'CallB'
            else: return 'CallC'
    if 'Callable' not in input_df.columns and 'Call Dt' in input_df.columns:
        input_df['Callable'] = input_df['Call Dt'].notna().map({True: 'Y', False: 'N'})
    input_df['Maturity-Call-Range'] = input_df.apply(
        lambda row: assign_name(pd.to_numeric(row.get('YrTMat'), errors='coerce'),
                                pd.to_numeric(row.get('Call Wndw'), errors='coerce'),
                                row.get('Callable','N')), axis=1
    )

    # Sector map (from App B)
    def map_sector(bloomberg_sector, bics_level_3_industry_name):
        if bloomberg_sector == "General Obligation":
            return "State GO" if bics_level_3_industry_name == "States & Territories" else "Local GO"
        if bloomberg_sector == "Bond Bank":
            return "Transportation" if bics_level_3_industry_name in ["Airports","Highways, Bridges & Tunnels"] else "Special Tax"
        if bloomberg_sector == "Loan Pool":
            return "Lease " if bics_level_3_industry_name == "States & Territories" else "Water & Sewer"
        local_go_sub = ['School District','General Revenue Tax-Guaranteed','Community College District','Municipal Utility District',
                        'General Obligation Hospital/Health District','Water & Sewer Tax-Guaranteed','General Obligation District (Other)',
                        'Parking Tax-Guaranteed','Housing Tax-Guaranteed','Healthcare (General) Tax-Guaranteed','Metro Development District','Airport Tax-Guaranteed']
        if bloomberg_sector in local_go_sub: return "Local GO"
        education_sub = ["Higher Education","Charter School","Secondary Education","Student Housing","Student Loan Revenue","Private/Religious School"]
        if bloomberg_sector in education_sub: return "Education"
        health_sub = ['Hospital','Continuing Care Retirement Community','Not-For-Profit Human Service Provider','Assisted Living']
        if bloomberg_sector in health_sub: return "Hospital"
        housing_sub = ['State Multi-Family Housing','State Single-Family Housing','Local Multi-Family Housing','Local Single-Family Housing','Mobile Home Housing']
        if bloomberg_sector in housing_sub: return "Housing"
        if bloomberg_sector == "Appropriation": return "Lease"
        power_sub = ['Public Power System','Municipal Utility (Mixed)','Nuclear Power']
        if bloomberg_sector in power_sub: return "Power"
        special_tax_sub = ['Income Tax','Loan Pool','Miscellaneous Tax','Hotel Occupancy Tax','Bond Bank','Appropriation (Self)','Sales & Excise Tax',
                           'Tax Increment Financing','Lottery','Mello-Roos','Special Assessment Financing','Telecom','Payments in Lieu of Taxes (PILOT)']
        if bloomberg_sector in special_tax_sub: return "Special Tax"
        if bloomberg_sector == "Tobacco Master Settlement Agreement": return "Tobacco Master Settlement Agreement"
        transportation_sub = ['Toll Highway/Bridge/Tunnel','Non-Toll Highway/Bridge/Tunnel','Airport','Public Transportation','Port/Marina',
                              'Grant Anticipation Revenue Vehicle','Parking Facility']
        if bloomberg_sector in transportation_sub: return "Transportation"
        water_sewer_sub = ['Water & Sewer','Solid Waste']
        if bloomberg_sector in water_sewer_sub: return "Water & Sewer"
        idr_pcr_sub = ['Economic/Industrial Development','Gas Forward Contract','Indian Tribal Bond','Not-For-Profit Cultural Organization',
                       'Not-For-Profit Foundation','Not-For-Profit Membership Organization','Not-For-Profit Research Organization',
                       'Payments in Lieu of Taxes (PILOT)']
        if bloomberg_sector in idr_pcr_sub: return "IDR/PCR"
    bb3_alias = get_first_existing_col(input_df, ['BB Sector 3','BICS Level 3 Industry Name'])
    if 'BB Sector' in input_df.columns and bb3_alias:
        input_df['Muni Sector'] = input_df.apply(lambda r: map_sector(r['BB Sector'], r[bb3_alias]), axis=1)
        if 'Prere' in input_df.columns:
            input_df.loc[input_df['Prere'] == 'Y', 'Muni Sector'] = 'Pre-Refunded'

    # Rating scores -> Highest Rating
    if 'Guaranty Agmt' in input_df.columns:
        input_df['GUARANTOR_NAME'] = ensure_str(input_df['Guaranty Agmt']).str.upper()

    # Weighted percentile
    def weighted_percentile(data: np.ndarray, weights: np.ndarray, q: float):
        order = np.argsort(data)
        data = data[order]; weights = weights[order]
        cum_w = np.cumsum(weights)
        cutoff = q * cum_w[-1]
        idx = np.searchsorted(cum_w, cutoff)
        return data[min(idx, len(data)-1)]

    results = []
    for _, row in input_df.iterrows():
        coupon_val = float(row['COUPON']) if pd.notna(row['COUPON']) else np.nan
        if np.isfinite(coupon_val) and coupon_val in [4.0, 5.0]:
            coupon_filter = (reference_df['COUPON'] == coupon_val)
        else:
            lower_bound = max(4.0, coupon_val - 0.51) if np.isfinite(coupon_val) else 4.0
            upper_bound = coupon_val + 0.5 if np.isfinite(coupon_val) else 999.0
            coupon_filter = reference_df['COUPON'].between(lower_bound, upper_bound)

        final_rating_val = int(row['Highest Rating']) if pd.notna(row.get('Highest Rating')) else None

        sinkable_row_flag = ensure_str(pd.Series([row.get('Sinkable','')])).iloc[0].upper() == 'Y'
        yrtmat_val = pd.to_numeric(row.get('YrTMat'), errors='coerce')
        sinkable_filter = True
        if sinkable_row_flag and pd.notna(yrtmat_val) and (yrtmat_val <= 15):
            ref_maturity = pd.to_numeric(reference_df['Maturity Years From Today'], errors='coerce')
            sinkable_filter = ((reference_df['Sinkable'].str.upper() == 'Y') & (ref_maturity <= 15))

        # STRICT exact sector match
        candidates = reference_df[
            (reference_df['Duration Category'] == row['Duration Category']) &
            (reference_df['AMT'] == row.get('AMT','N')) &
            (reference_df['Muni Sector'] == row['Muni Sector']) &   # exact match required
            (reference_df['Final Rating'].between(final_rating_val - 1, final_rating_val + 1)) &
            coupon_filter &
            (reference_df['Maturity-Call-Range'] == row['Maturity-Call-Range']) &
            (reference_df['Putable'] == row.get('Putable','N')) &
            sinkable_filter
        ].copy()

        # GUARANTOR filter for IDR/PCR
        if (row.get('Muni Sector') == 'IDR/PCR' and
            str(row.get('GUARANTOR_NAME','')).strip() != '' and
            ('GUARANTOR_NAME' in candidates.columns)):
            guarantor_key = str(row['GUARANTOR_NAME']).strip().upper()
            candidates = candidates[candidates['GUARANTOR_NAME'] == guarantor_key]

        state_val = row.get('STATE','')
        restricted_states = ['CA','NY','NJ']
        if state_val in restricted_states:
            candidates = candidates[candidates['STATE'] == state_val]
        else:
            candidates = candidates[~candidates['STATE'].isin(restricted_states)]

        base_fields = {
            'CUSIP': row.get('Cusip') or row.get('CUSIP'),
            'Duration Category': row.get('Duration Category'),
            'Muni Sector': row.get('Muni Sector'),
            'Maturity-Call-Range': row.get('Maturity-Call-Range'),
            'Putable': row.get('Putable','N'),
            'Final Rating': final_rating_val,
            'YIELD': row.get('YIELD', np.nan),
            'BVAL Yld': pd.to_numeric(row.get('BVAL Yld', np.nan), errors='coerce'),
            'Call Wndw': row.get('Call Wndw', np.nan)
        }

        if not candidates.empty:
            gs = pd.to_numeric(candidates['G_SPREAD'], errors='coerce').dropna()
            if gs.empty:
                results.append({**base_fields,'Median_GSpread':None,'P80_GSpread':None,'Max_GSpread':None,'Max_Plus_Gap75to100':None,'Observations':0})
                continue

            if state_val in ["TX","IL","FL"]:
                candidates['Weight'] = candidates['STATE'].apply(lambda s: 1.3 if s == state_val else 1)
            else:
                candidates['Weight'] = 1

            weights = candidates.loc[gs.index, 'Weight'].to_numpy()
            gs_vals = gs.to_numpy()

            def wperc(q): return weighted_percentile(gs_vals, weights, q)
            median_gspread = wperc(0.50)
            p80_gspread    = wperc(0.80)
            max_gspread    = float(np.max(gs_vals))
            max_plus_gap   = (2 * max_gspread - p80_gspread)
            obs_count      = len(gs_vals)

            results.append({
                **base_fields,
                'Median_GSpread': round(median_gspread, 2) if np.isfinite(median_gspread) else None,
                'P80_GSpread': round(p80_gspread, 2) if np.isfinite(p80_gspread) else None,
                'Max_GSpread': round(max_gspread, 2) if np.isfinite(max_gspread) else None,
                'Max_Plus_Gap75to100': round(max_plus_gap, 2) if np.isfinite(max_plus_gap) else None,
                'Observations': obs_count
            })
        else:
            results.append({**base_fields,'Median_GSpread':None,'P80_GSpread':None,'Max_GSpread':None,'Max_Plus_Gap75to100':None,'Observations':0})

    output_df = pd.DataFrame(results)

    valid_metrics = ['Median_GSpread','P80_GSpread','Max_GSpread','Max_Plus_Gap75to100']
    if metric_option not in valid_metrics:
        metric_option = 'Median_GSpread'

    display_df = output_df[['CUSIP','Duration Category','Muni Sector','Maturity-Call-Range','Putable','Final Rating','YIELD',metric_option,'Observations','BVAL Yld']].copy()
    if adjustment_bp != 0:
        display_df[metric_option] = display_df[metric_option].apply(lambda x: round(x + adjustment_bp, 2) if pd.notna(x) else None)

    display_df['Bid Yield'] = (pd.to_numeric(display_df['YIELD'], errors='coerce') + (pd.to_numeric(display_df[metric_option], errors='coerce') / 100.0)).round(3)
    display_df['BVAL Yld'] = pd.to_numeric(display_df['BVAL Yld'], errors='coerce')
    display_df['ReplacedWithBVAL'] = False
    mask = (display_df['Bid Yield'] < display_df['BVAL Yld'])
    display_df.loc[mask, 'Bid Yield'] = display_df.loc[mask, 'BVAL Yld']
    display_df.loc[mask, 'ReplacedWithBVAL'] = True

    return display_df

# -------------------------------------------------
# UI (single-button run)
# -------------------------------------------------
pasted = st.text_area(
    "Paste tab‑separated BWIC data (include headers):",
    height=220,
    placeholder="Cusip\tPar\tIssuer\tEff Dur\tCpn\tYrTMat\tCall Dt\tMat Dt\tPut Dt\tSinkable\tPutable\tPrere\tSt\tBB Sector\tBB Sector 3\tBVAL Yld\tInsured\tAMT\t144A\tTaxable"
)

# Core controls
colA, colB, colC = st.columns([1,1,1])
with colA:
    include_ca   = st.checkbox("Include CA", value=False)
    include_ny   = st.checkbox("Include NY", value=True)
    st.markdown("**Duration buckets**")
    duration_options    = ["0-1","1-2","2-3","3-6","6-10","10+"]
    default_durations   = ["2-3","3-6","6-10","10+"]
    selected_durations  = [opt for opt in duration_options if st.checkbox(opt, value=opt in default_durations, key=f"dur_{opt}")]
with colB:
    include_prere = st.checkbox("Include Pre‑refunded", value=False)
    start_year, end_year = st.slider("Maturity Year Range", 2025, 2060, (2025, 2060))
    st.markdown("**Ratings**")
    rating_options   = ["AAA","AA","A","BBB","HY"]
    default_ratings  = ["AAA","AA","A","BBB"]
    selected_ratings = [opt for opt in rating_options if st.checkbox(opt, value=opt in default_ratings, key=f"rt_{opt}")]
with colC:
    coupon_lower = st.number_input("Coupon ≥", min_value=0.0, max_value=10.0, value=4.0, step=0.25)
    coupon_upper = st.number_input("Coupon ≤", min_value=0.0, max_value=10.0, value=5.0, step=0.25)
    st.markdown("**Exclude Guarantors (Prepaid Gas)**")
    guarantor_options = ['Athene','DB','MS','GS','RBC','Citi','PacLife','TD','JPM','BP']
    default_guarantors = ['Athene','DB']
    guarantors_vars    = {opt: st.checkbox(opt, value=opt in default_guarantors, key=f"g_{opt}") for opt in guarantor_options}

st.markdown("---")
metric_display_names = {
    "Median": "Median_GSpread",
    "80th Percentile": "P80_GSpread",
    "Maximum": "Max_GSpread",
    "Max + Gap (75th→100th)": "Max_Plus_Gap75to100"
}
metric_option_display = st.selectbox("Spread metric", list(metric_display_names.keys()), index=0)
metric_option = metric_display_names[metric_option_display]

market_env = st.selectbox("Market environment (bps)", options=["Normal","+10","+20","+30"], index=0)
adjustment_bp = {"Normal":0, "+10":10, "+20":20, "+30":30}[market_env]

# ONE BUTTON
run = st.button("▶️ Run Filters and Generate Bids", use_container_width=True)

if run:
    if not pasted.strip():
        st.warning("Please paste your tab‑separated BWIC data first.")
    else:
        try:
            # Input
            df_raw = pd.read_csv(io.StringIO(pasted), sep="\t")
            if 'Cusip' not in df_raw.columns and 'CUSIP' in df_raw.columns:
                df_raw = df_raw.rename(columns={'CUSIP':'Cusip'})

            # App A
            filtered_full = filter_data_app_a(
                df=df_raw,
                include_ca=include_ca,
                include_ny=include_ny,
                include_prere=include_prere,
                start_year=start_year,
                end_year=end_year,
                coupon_lower=coupon_lower,
                coupon_upper=coupon_upper,
                selected_ratings=selected_ratings,
                selected_durations=selected_durations,
                guarantors_vars=guarantors_vars,
            )
            if len(filtered_full) == 0:
                st.warning("No rows after App A filters. Widen your year/coupon ranges or untick rating/duration/guarantor exclusions.")
                st.stop()

            # App B
            reference_df, curve_data = load_reference_tables()
            reference_df = prepare_reference_df(reference_df)

            app_b_display = compute_gspread_display(
                input_df=filtered_full,
                reference_df=reference_df,
                curve_data=curve_data,
                metric_option=metric_option,
                adjustment_bp=adjustment_bp
            )

            
            # Merge App A context (CUSIP)
            app_a_context_cols = [c for c in [
                'Cusip','Par','Issuer','Days Since Rating Action','ERP',
                'Final Rating','Duration Category','Muni Sector'
            ] if c in filtered_full.columns]
            
            app_a_context = filtered_full[app_a_context_cols].copy()
            if 'Cusip' in app_a_context.columns:
                app_a_context = app_a_context.rename(columns={'Cusip':'CUSIP'})
            
            final_df = app_b_display.merge(app_a_context, on='CUSIP', how='left')

            
            cols_order = ['CUSIP','Par','Issuer','Days Since Rating Action','ERP',
                          'Bid Yield','BVAL Yld','Observations']
            final_df = final_df[[c for c in cols_order if c in final_df.columns]]

            
            # Show all columns including Days Since Rating Action
            st.success("✅ Bonds Filtered and Bids Generated")
            st.dataframe(final_df, use_container_width=True)

            csv_data = final_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download Unified Output (CSV)",
                data=csv_data,
                file_name="combined_muni_bwic_output.csv",
                mime="text/csv"
            )

        except Exception as e:
            st.error(f"Error processing data: {e}")
else:
    st.warning("Please paste your data first.")
