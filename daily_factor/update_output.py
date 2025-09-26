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
def save_predictions_to_excel(predictions, test_data, output_dir="result"):
    os.makedirs(output_dir, exist_ok=True)
    today_str = datetime.today().strftime("%Y%m%d")
    output_file = os.path.join(output_dir, f"{today_str}-predict.xlsx")

    with pd.ExcelWriter(output_file) as writer:
        for pred_label, preds in predictions.items():
            # 构造结果表：包含真实值和预测
            df_out = pd.DataFrame({
                "date": test_data.index,                  # 日期
                pred_label: test_data[pred_label],        # 对应的特征值
                "label": test_data["label"],              # 真实标签
                "prediction": preds                       # 模型预测
            })
            # sheet 名直接用 pred_label
            df_out.to_excel(writer, sheet_name=pred_label, index=False)
    print(f"预测结果已保存到 {output_file}")


if __name__ == "__main__":
    # 是否重新训练模型，没有GPU会很慢，一个模型大概50分钟（如果采用best参数）
    # 如果不重新训练，则直接加载已有模型进行预测
    # 重新训练时，模型会保存在 AutogluonModels 文件夹下
    # 参数具体信息详见AutoGluon文档
    retrain = False  
    if retrain == False:
        pred_labels = ['basis__TL', 'basis__T', 'basis__TF', 'basis__TS']
        start_date = '2025/09/25'  # 指定预测开始的日期，不要早于训练集的结束日期
        data.run_data()
        factor.run_factor()
        myfactor = nf.NewFactor()
        myfactor.get_data('basis').to_excel('factor/basis.xlsx')
        df_raw = merge_excels_by_first_col('factor', 'merged.xlsx')
        df_raw.set_index('date', inplace=True)
        predictions = {}
        # 涨跌标签：下一期比当前期涨为 1，跌为 -1，不变为 0
        for pred_label in pred_labels:
            df_clean = df_raw.copy()
            df_clean['label'] = (df_clean[pred_label].shift(-1) - df_clean[pred_label]).apply(
                lambda x: 1 if x > 0 else -1 if x < 0 else 0
            )
            df_clean = df_clean.iloc[:-1]
            predictor = TabularPredictor.load("AutogluonModels\\best_" + pred_label)
            test_data = df_clean[df_clean.index > start_date]
            print(f"Evaluating model for {pred_label} on test data after {start_date}:")
            predictions[pred_label] = predictor.predict(test_data)
        save_predictions_to_excel(predictions, test_data)
        
