import data
import new_factor as nf
import numpy as np
import pandas as pd
from autogluon.tabular import TabularDataset, TabularPredictor
import factor
import re
import os, sys
from datetime import datetime

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
def save_predictions_to_excel(predictions, test_data_map, output_dir="result"):
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

            # 写入各自 sheet
            df_out.to_excel(writer, sheet_name=pred_label, index=False)

    print(f"预测结果已保存到 {output_file}")



if __name__ == "__main__":
    # 是否重新训练模型，没有GPU会很慢，一个模型大概50分钟（如果采用best参数）
    # 如果不重新训练，则直接加载已有模型进行预测
    # 重新训练时，模型会保存在 AutogluonModels 文件夹下
    # 参数具体信息详见AutoGluon文档
    retrain = False  
    if retrain == False:
        pred_labels = ['basis__TL', 'basis__T', 'basis__TF', 'basis__TS'] # 预测目标列名列表
        start_date = pd.to_datetime('2025-07-25')  # 设置预测起始日期
        data.run_data()
        factor.run_factor()
        myfactor = nf.NewFactor()
        myfactor.get_data('basis').to_excel('factor/basis.xlsx')
        df_raw = merge_excels_by_first_col('factor', 'merged.xlsx')
        
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
            df_clean['label'] = (df_clean[pred_label].shift(-1) - df_clean[pred_label]).apply(
                lambda x: 1 if x > 0 else -1 if x < 0 else 0
            )
            df_clean = df_clean.iloc[:-1]

            # ========== 预测 ==========
            predictor = TabularPredictor.load("AutogluonModels\\best_" + pred_label)
            test_data = df_clean[df_clean.index > start_date]
            print(f"Evaluating model for {pred_label} on test data from {start_date}: rows={len(test_data)}")

            if len(test_data) == 0:
                print(f"[WARN] {pred_label} 在 {start_date} 及之后没有数据，跳过预测与保存。")
                continue

            preds = predictor.predict(test_data)
            predictions[pred_label] = preds
            test_data_map[pred_label] = test_data

        save_predictions_to_excel(predictions, test_data_map)
        
