import data
import factor
import time
import importlib
importlib.reload(data)
importlib.reload(factor)
if __name__ == "__main__":
    date_end = int(time.strftime('%Y%m%d'))
    data.update_data(date_end)
    factor.update_factor(date_end)
    print('因子更新完成, 日期: ', date_end)