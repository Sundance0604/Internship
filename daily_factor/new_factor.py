# -*- coding: utf-8 -*-
import time
import re
import pandas as pd
from WindPy import w
import os
import openpyxl

class NewFactor:
    """
    - get_data(bonds_name, ...): 从 Wind 取某一类曲线的数据（如 'national'、'CDB'）
    - get_spread(bonds_name, ...): 计算同一类曲线的相邻期限利差（如 国债10Y-国债7Y）
    - get_spread_by_codes(bonds1, bonds2, ...): 计算两类曲线在相同后缀(如 '1Y')下的利差（如 国债10Y-国开10Y）
    """

    def __init__(self, start_date=20170102, end_date=None, auto_connect=True):
        # 代码和列名的映射
        self.codes = {
            'national': ['M1000158','M1000159','M1000160','M1000162','M1000164','M1000163','M1000166','M1000170'],
            'CDB'     : ['M1004263','M1004264','M1004265','M1004267','M1004269','M1004271','M1004273','M1004275'],
            'FR007'   : ['M0048484','M0048485','M0048486','M0048487','M0048490'],
            'SHIBOR'  : ['M0048499','M0048500','M0048501','M0075930'],
            'perpetual':["R2314607","A1228046","J0232832","Z9025043","X3441834","R8982978"],
            'tier2'   :["M1010704","M1010705","M1010706","M1010708","M1010710","M1010713"],
            'NCD':      ['M1004902'],
            'basis':  ['A9301800','R3689129','R9310826','A8354326']
        }
        self.names = {
            'national': ['国债_1Y','国债_2Y','国债_3Y','国债_5Y','国债_7Y','国债_10Y','国债_30Y','国债_50Y'],
            'CDB'     : ['国开债_1Y','国开债_2Y','国开债_3Y','国开债_5Y','国开债_7Y','国开债_10Y','国开债_30Y','国开债_50Y'],
            'FR007'   : ['FR007_6M','FR007_9M','FR007_1Y','FR007_2Y','FR007_5Y'],
            'SHIBOR'  : ['SHIBOR_1Y','SHIBOR_2Y','SHIBOR_3Y','SHIBOR_5Y'],
            'perpetual':['永续债_1Y','永续债_2Y','永续债_3Y','永续债_5Y','永续债_7Y','永续债_10Y'],
            'tier2'   : ['二级资本债_1Y','二级资本债_2Y','二级资本债_3Y','二级资本债_5Y','二级资本债_7Y','二级资本债_10Y'],
            'NCD' :     ['同业存单_1Y'],
            'basis': ['T','TF','TS','TL']
        }

        # 日期参数
        self.start_date = start_date
        self.end_date   = end_date if end_date is not None else int(time.strftime('%Y%m%d'))

        # 简单缓存：key = (bonds_name, start_int, end_int)
        self.cache = {}

        # 保存路径
        self.path = "D:\\mycodelife\\workshop\\yinhe\\factor_data"
        # Wind 连接
        if auto_connect:
            w.start()
            if not w.isconnected():
                raise RuntimeError("WindPy 未连接成功，请检查 Wind 客户端或网络。")

    # --------- 工具函数 ---------
    @staticmethod
    def _int_to_date(int_yyyymmdd):
        """int(YYYYMMDD) -> pandas.Timestamp(yyyy-mm-dd 00:00:00)"""
        return pd.to_datetime(str(int_yyyymmdd), format="%Y%m%d")

    @staticmethod
    def _norm_index_to_date(dfx):
        """将 index 转为无时区、仅日期（normalize）"""
        dfx.index = pd.to_datetime(dfx.index)
        if getattr(dfx.index, "tz", None) is not None:
            dfx.index = dfx.index.tz_localize(None)
        dfx.index = dfx.index.normalize()
        return dfx

    @staticmethod
    def _suffix(col):
        """提取列名后缀（支持 1Y/2Y/6M 等），例如 国债_10Y -> 10Y"""
        m = re.search(r'_((\d+Y)|(\d+M))$', col)
        return m.group(1) if m else None

    # --------- 数据获取 ---------
    def get_data(self, bonds_name, date_start=None, date_end=None, freq="B", fill=False):
        """
        拉取某类品种的时间序列
        - bonds_name: 'national'/'CDB'/...
        - date_start/date_end: int(YYYYMMDD) 或 None（默认用 self.start/end_date）
        - freq: 目标索引频率，默认工作日 'B'
        - fill: 是否用前值填充空缺
        """
        # 规范化日期参数
        start_i = date_start if date_start is not None else self.start_date
        end_i   = date_end   if date_end   is not None else self.end_date
        start_dt = self._int_to_date(start_i)
        end_dt   = self._int_to_date(end_i)

        # 缓存命中
        cache_key = (bonds_name, start_i, end_i, freq, fill)
        if cache_key in self.cache:
            return self.cache[cache_key].copy()

        # Wind 取数
        frames = []
        start_str = start_dt.strftime("%Y-%m-%d")
        end_str   = end_dt.strftime("%Y-%m-%d")
        for col, code in zip(self.names[bonds_name], self.codes[bonds_name]):
            try:
                # usedf=True: 返回 (ErrorCode, DataFrame)
                err, res = w.edb(code, start_str, end_str, usedf=True)
                if err != 0:
                    # Wind 返回错误码时给空框架
                    res = pd.DataFrame(index=pd.DatetimeIndex([], name="date"))
            except Exception:
                res = pd.DataFrame(index=pd.DatetimeIndex([], name="date"))

            if res is None or len(res) == 0:
                # 构造空框架，保持列名
                res = pd.DataFrame(index=pd.date_range(start_dt, end_dt, freq=freq), columns=[col])
            else:
                res = res.copy()
                # 规范索引与列名
                res.index = pd.to_datetime(res.index).tz_localize(None).normalize()
                res.rename(columns={res.columns[0]: col}, inplace=True)
                # 重采样/重建索引到工作日
                res = res.reindex(pd.date_range(start_dt, end_dt, freq=freq))

            if fill:
                res = res.ffill()

            frames.append(res)

        data = pd.concat(frames, axis=1)

        # 写入缓存
        self.cache[cache_key] = data.copy()
        return data

    # --------- 同类期限利差 ---------
    def get_spread(self, bonds_name, df=None, start_date=None, end_date=None):
        """
        计算同一类曲线的相邻期限利差：names[i] - names[i-1]
        注意：返回的是复制后的 DataFrame，不会污染缓存
        """
        if df is None:
            df = self.get_data(bonds_name, start_date, end_date)
        out = df.copy()

        names = self.names[bonds_name]
        for i in range(1, len(names)):
            a, b = names[i], names[i-1]
            if a in out.columns and b in out.columns:
                out[f"{a}-{b}"] = out[a] - out[b]
        return out

    # --------- 跨类相同后缀利差 ---------
    def get_spread_by_codes(self, bonds1, bonds2, df1=None, df2=None,
                            start_date=None, end_date=None, freq="B", fill=False,
                            strict_pair=False):
        """
        计算两类曲线在相同后缀(如 1Y/2Y/...) 下的利差：
        例如 国债_10Y - 国开债_10Y
        - freq: 对齐到的索引频率（默认工作日）
        - fill: 是否用前值填充空缺
        - strict_pair: 若某后缀一侧多列一侧少列，True 则报错；False（默认）按最短配对并给出告警
        """
        # 取数 / 复制
        if df1 is None:
            df1 = self.get_data(bonds1, start_date, end_date, freq=freq, fill=fill)
        else:
            df1 = df1.copy()
        if df2 is None:
            df2 = self.get_data(bonds2, start_date, end_date, freq=freq, fill=fill)
        else:
            df2 = df2.copy()

        # 统一索引到日期
        self._norm_index_to_date(df1)
        self._norm_index_to_date(df2)

        # 取行索引交集
        idx = df1.index.intersection(df2.index)
        df1 = df1.loc[idx]
        df2 = df2.loc[idx]

        # 后缀映射与交集
        map1 = {c: self._suffix(c) for c in df1.columns}
        map2 = {c: self._suffix(c) for c in df2.columns}
        common = set(filter(None, map1.values())) & set(filter(None, map2.values()))
        # 仅保留有共同后缀的列
        df1 = df1[[c for c in df1.columns if map1[c] in common]]
        df2 = df2[[c for c in df2.columns if map2[c] in common]]

        # 构造配对
        pairs, warnings = [], []
        for suf in sorted(common):
            c1 = sorted([c for c in df1.columns if map1[c] == suf])
            c2 = sorted([c for c in df2.columns if map2[c] == suf])
            if len(c1) != len(c2):
                msg = f"后缀 {suf}: 左{len(c1)}列 vs 右{len(c2)}列"
                if strict_pair:
                    raise ValueError(msg + "（strict_pair=True）")
                warnings.append(msg + "，按最短配对，多余列忽略")
            for a, b in zip(c1, c2):
                pairs.append((a, b))

        # 做差
        out = pd.DataFrame(index=idx)
        for a, b in pairs:
            out[f"{a}-{b}"] = df1[a] - df2[b]

        # 输出前可选再对齐到完整工作日日历（起止按 idx 范围）
        # 若已经在 get_data 中用 freq='B' 拉过，这里通常不需要再次 reindex。
        if fill:
            out = out.ffill()

        if warnings:
            print("\n".join(warnings))
        return out

    def merge_data(self, path=None):
        
        path = self.path if path is None else path

        xlsx_files = [f for f in os.listdir(path) if f.endswith(".xlsx")]
        if not xlsx_files:
            raise FileNotFoundError("目录中没有找到 .xlsx 文件")

        merged_df = None
        for file in xlsx_files:
            file_path = os.path.join(path, file)
            df = pd.read_excel(file_path, index_col=0)  # 用第一列作为 index
            df.index = pd.to_datetime(df.index, errors="ignore")  # 如果是日期可以解析成日期型

            if merged_df is None:
                merged_df = df
            else:
                # 按 index 全连接
                merged_df = merged_df.merge(df, left_index=True, right_index=True, how="outer")
        merged_df.to_excel(f"{path}/全因子.xlsx")

    def save_data(self, myfactor):
        myfactor.get_data('national').to_excel("国债.xlsx")
        print("已保存")
        myfactor.get_spread('national').to_excel('国债期限利差.xlsx')
        myfactor.get_spread_by_codes('CDB','national').to_excel("国债国开利差.xlsx")
        myfactor.get_spread_by_codes('perpetual','tier2').to_excel("永续二级利差.xlsx")
        myfactor.get_spread_by_codes('national','national').to_excel("永续国债利差.xlsx")
        myfactor.get_spread_by_codes('tier2','CDB').to_excel("二级国开利差.xlsx")
        myfactor.get_spread_by_codes('SHIBOR','FR007').to_excel("SHIBOR FR007利差.xlsx")
        myfactor.get_spread_by_codes('national','FR007').to_excel("FR007 国债利差.xlsx")
        myfactor.get_spread_by_codes('NCD','FR007').to_excel("FR007 同业存单.xlsx")
if __name__ == "__main__":
    myfactor = NewFactor()
    # myfactor.save_data(myfactor)
    # myfactor.merge_data() 
    myfactor.get_data('FR007').to_excel("FR007.xlsx")

