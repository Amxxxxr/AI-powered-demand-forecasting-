"""Run the whole pipeline in order.   python run_pipeline.py [--skip-lstm] [--skip-eda]"""
import sys
from src import data_preparation, eda, train_baseline, train_xgboost, train_lstm, evaluate

steps = [("1/6 Data preparation", data_preparation.main)]
if "--skip-eda" not in sys.argv: steps.append(("2/6 EDA", eda.main))
steps += [("3/6 Baseline", train_baseline.main), ("4/6 XGBoost", train_xgboost.main)]
if "--skip-lstm" not in sys.argv: steps.append(("5/6 LSTM", train_lstm.main))
steps.append(("6/6 Evaluation", evaluate.main))
for name, fn in steps:
    print(f"\n===== {name} =====")
    fn()
print("\nPipeline finished. Start the app with:  python -m backend.app")
