import datetime

from inference import inference
from prepare_data import prepare_data

if __name__ == '__main__':
    time_series_data = prepare_data()
    start_date = datetime.date(2019, 3, 16)
    
    inference(time_series_data, start_date, save_output=True)
