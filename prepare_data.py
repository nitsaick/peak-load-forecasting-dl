import pandas as pd


def prepare_data():
    df = pd.read_csv('./data/holiday.csv', parse_dates=[0], encoding='UTF-8-sig', engine='python')
    df['holiday'] = pd.to_datetime(df['holiday']).dt.date
    holiday = df['holiday'].values

    df = pd.read_csv('./data/workday.csv', parse_dates=[0], encoding='UTF-8-sig', engine='python')
    df['workday'] = pd.to_datetime(df['workday']).dt.date
    workday = df['workday'].values

    df = pd.read_csv('./data/台灣電力公司_過去電力供需資訊.csv', parse_dates=[0], encoding='UTF-8-sig', engine='python')
    df.rename(columns={'尖峰負載(MW)': 'peak_load', '日期': 'date'}, inplace=True)
    df['weekday'] = df['date'].dt.dayofweek
    df['workday'] = ~df['weekday'].isin([5, 6]) & ~df['date'].isin(holiday) | \
                    df['weekday'].isin([5, 6]) & df['date'].isin(workday)

    magic_num = 50000
    df['workday'] *= magic_num
    df = df.set_index('date')

    data = df[['peak_load', 'workday']]
    return data
