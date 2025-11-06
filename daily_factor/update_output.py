import data
import new_factor as nf
import numpy as np
import pandas as pd
from autogluon.tabular import TabularDataset, TabularPredictor
import factor
import re
import os, sys
from datetime import datetime
import pathlib
import platform

# 如果是在 Windows 上运行，从 Linux 保存的模型加载：
if platform.system() == "Windows":
    # 把 PosixPath 替换成 WindowsPath，防止 pickle 报错
    pathlib.PosixPath = pathlib.WindowsPath

# 把脚本所在目录设为当前工作目录
os.chdir(os.path.dirname(os.path.abspath(__file__)))

def merge_excels_by_first_col(folder_path, output_file="merged.xlsx"):
    # 排除：以8位日期开头、紧跟“-因子清单”的文件（不含扩展名匹配）
    pattern = re.compile(r"^\d{8}-因子清单")
    files = [
        f for f in os.listdir(folder_path)
        if f.endswith(".xlsx")
        and not f.startswith("~")
        and not pattern.match(os.path.splitext(f)[0])
    ]
    if not files:
        print("未找到任何可用的 .xlsx 文件")
        return
    dfs = []
    for f in sorted(files):
        path = os.path.join(folder_path, f)
        try:
            # 第一行是列名，第一列作为索引（合并键）
            df = pd.read_excel(path, header=0, index_col=0)
        except Exception as e:
            print(f"读取失败：{path}，原因：{e}")
            continue
        # 规范化索引（第一列），避免空白/类型不一致
        df.index = df.index.astype(str).str.strip()
        # 列名加前缀（保留原始列名，避免冲突）
        stem = os.path.splitext(f)[0]
        df.columns = [f"{stem}__{col}" for col in df.columns]
        dfs.append(df)
    if not dfs:
        print("没有成功读取的表格")
        return
    # 以索引（第一列）按“交集”对齐拼接
    merged = pd.concat(dfs, axis=1, join="inner")
    # 索引转为普通列，并命名为 "date"
    merged = merged.reset_index().rename(columns={"index": "date"})
    # 导出
    merged.to_excel(output_file, index=False, header=True)
    return merged
# 训练模型时调用
def data_split(df_clean):
    split_point = int(len(df_clean) * 0.8)
    # 按顺序切分
    df_train = df_clean.iloc[:split_point]
    df_test  = df_clean.iloc[split_point:]
    return df_train, df_test
# 保存预测结果到 Excel
def save_predictions_to_excel(predictions, test_data_map, output_dir="result", inverse=False):
    os.makedirs(output_dir, exist_ok=True)
    today_str = datetime.today().strftime("%Y%m%d")
    output_file = os.path.join(output_dir, f"{today_str}-predict.xlsx")

    with pd.ExcelWriter(output_file) as writer:
        for pred_label, preds in predictions.items():
            td = test_data_map.get(pred_label, None)
            if td is None or td.empty:
                # 没有数据就写个空壳，避免报错
                pd.DataFrame(columns=["date", pred_label, "label", "prediction"]).to_excel(
                    writer, sheet_name=pred_label, index=False
                )
                continue

            # 构造结果表：包含真实值、特征值和预测
            df_out = pd.DataFrame({
                "date": td.index,            # 日期（索引）
                pred_label: td[pred_label],  # 对应特征列
                "label": td["label"],        # 真实标签
            })

            # 对齐预测结果（preds 通常为 Series，索引与 td 一致；稳妥起见重建为按 td.index 对齐）
            preds_aligned = pd.Series(preds, index=td.index)
            df_out["prediction"] = preds_aligned.values
            if inverse:
                df_out["prediction"] = -df_out["prediction"]
                df_out["label"] = -df_out["label"]
            # 写入各自 sheet
            df_out.to_excel(writer, sheet_name=pred_label, index=False)

    print(f"预测结果已保存到 {output_file}")

def replace_any_date(df: pd.DataFrame, new_date: str) -> pd.DataFrame:
    """
    替换列名中任意位置的 8 位日期（如 20250924）为指定的新日期字符串。
    参数:
        df (pd.DataFrame): 原始 DataFrame。
        new_date (str): 新日期字符串，格式为 'YYYYMMDD'（8位数字）。
    返回:
        pd.DataFrame: 替换列名后的 DataFrame。
    """
    pattern = re.compile(r'\b\d{8}\b')  # 匹配独立的8位数字，避免误替非日期数字片段

    new_columns = [pattern.sub(new_date, col) for col in df.columns]
    df_renamed = df.copy()
    df_renamed.columns = new_columns
    return df_renamed

def delete_files_in_directory(directory_path):
    """
    删除指定路径下的所有文件，但不删除文件夹。
    """
    # 遍历目录下的所有文件
    for filename in os.listdir(directory_path):
        file_path = os.path.join(directory_path, filename)
        if os.path.isfile(file_path):
            os.remove(file_path)  

if __name__ == "__main__":
    # 是否重新训练模型，没有GPU会很慢，一个模型大概50分钟（如果采用best参数）
    # 如果不重新训练，则直接加载已有模型进行预测
    # 重新训练时，模型会保存在 AutogluonModels 文件夹下
    # 参数具体信息详见AutoGluon文档
    models = {
        '基差':['basis__TL', 'basis__T', 'basis__TF', 'basis__TS'],
        'tier2':['tier2__二级资本债_5Y'],
        'FR007':['FR007__FR007_1Y', 'FR007__FR007_5Y'],
        '国债':['national__国债_10Y'],
        '国开债':['CDB__国开债_10Y'],
        '黄金':['gold__收盘价']
    }
    label_date = {
        '基差':'20250924',
        'tier2':['20251017','20240603'],
        'FR007':['20251016','20240108']
    }
    inverse_labels = ['基差', 'tier2', 'FR007', '黄金']  # 这些指标的标签取反
    inverse = True # 基差、二级资本债、FR007，黄金的指标取反（训练时搞忘了）
    add_result = True  # 已有因子数据时，更新因子数据（不重新取数）
    target = 'FR007'  # 选择需要预测的目标
    pred_labels = models[target] # 预测目标列名列表
    old_file_path = 'merged_FR007_20251029.xlsx'  # 旧的合并文件路径，用于追加结果
    start_date = pd.to_datetime(label_date[target][1])  # 这是在训练模型时测试集开始的时间
    today_str = datetime.today().strftime("%Y%m%d")
    delete_files_in_directory('factor')  # 清空因子文件夹，防止旧文件干扰合并
    if add_result: 
        old_merge_file = pd.read_excel(old_file_path)
        last_date = old_merge_file['date'].max()
        last_date = last_date.replace('-', '')
        last_date = str(int(last_date) + 1)
        print("last_date =", last_date)
        factor.run_factor(date_start=last_date)
        myfactor = nf.NewFactor(start_date=last_date)
    else:
        # data.run_data() # 第一次运行需要
        factor.run_factor(date_end="20251016")
        myfactor = nf.NewFactor(end_date="20251016")
    myfactor.get_data(target).to_excel(f'factor/{target}.xlsx')
    df_raw = merge_excels_by_first_col('factor', f'merged_{target}_{today_str}.xlsx')
    df_raw = replace_any_date(df_raw, label_date[target][0])
    if add_result:
        df_raw = pd.concat([old_merge_file, df_raw], ignore_index=True)
        df_raw.to_excel(f'merged_{today_str}.xlsx', index=False)
    if isinstance(df_raw.index, pd.DatetimeIndex):
        df_raw = df_raw.reset_index()

    # 统一列名
    if 'date' not in df_raw.columns:
        df_raw.rename(columns={df_raw.columns[0]: 'date'}, inplace=True)

    # 1) 先当字符串日期解析
    df_raw['date'] = pd.to_datetime(df_raw['date'], errors='coerce')

    df_raw = df_raw.dropna(subset=['date']).copy()
    df_raw = df_raw.sort_values('date').set_index('date')

    print("df_raw.index.min() =", df_raw.index.min())
    print("df_raw.index.max() =", df_raw.index.max())
    print("start_date         =", start_date)
    predictions = {}
    test_data_map = {}

    for pred_label in pred_labels:
        df_clean = df_raw.copy()

        for base_col in pred_labels:
            if base_col in df_clean.columns:
                df_clean[f'{base_col}_diff'] = df_clean[base_col].diff()
            else:
                print(f"[WARN] {base_col} 不存在于 df_clean.columns")

        # ========== 生成标签 ==========
        if inverse:
            df_clean['label'] = (df_clean[pred_label].shift(-1) - df_clean[pred_label]).apply(
                lambda x: 1 if x > 0 else -1 if x < 0 else 0
            )
        else:
            df_clean['label'] = (df_clean[pred_label].shift(-1) - df_clean[pred_label]).apply(
                lambda x: -1 if x > 0 else (1 if x < 0 else 0)
            )
            # 若为 0，则继承前一个标签
            df_clean['label'] = df_clean['label'].replace(0, method='ffill')
        df_clean = df_clean.iloc[:-1]

        # ========== 预测 ==========
        predictor = TabularPredictor.load("AutogluonModels\\best_" + pred_label, require_py_version_match=False)
        test_data = df_clean[df_clean.index > start_date]
        print(f"Evaluating model for {pred_label} on test data from {start_date}: rows={len(test_data)}")

        if len(test_data) == 0:
            print(f"[WARN] {pred_label} 在 {start_date} 及之后没有数据，跳过预测与保存。")
            continue

        preds = predictor.predict(test_data)
        predictions[pred_label] = preds
        test_data_map[pred_label] = test_data

    save_predictions_to_excel(predictions, test_data_map, inverse=inverse)
        
