import jsonlines
import numpy as np
import pandas as pd
import warnings
import argparse
warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser()
parser.add_argument('--result-path', help='Path to the JSON file of GPT evaluation results', type=str, required=True)
parser.add_argument('--model-name', help='Model name of the specific result given in --result-path', type=str, required=True)
args = parser.parse_args()

res = list(jsonlines.open(args.result_path))
model = args.model_name

all_res = {"consistency": [], "rejection": [], "knowledge": [], "lang": []}
for each in res:
    all_res['consistency'] += each['consist_scores']
    all_res['rejection'] += [int(p==t) for p,t in zip(each['reject_scores'], each['meta']['rejection_labels'])]
    all_res['knowledge'] += [s if s else np.nan for s in each['knowledge_scores']]
    all_res['lang'] += [each['meta']['lang']] * len(each['consist_scores'])
all_res_df = pd.DataFrame(all_res)
score_avg = ['all'] + list(all_res_df.mean(skipna=True))

score_lang_avg = all_res_df.groupby('lang').mean().reset_index()
all_score = score_lang_avg.append(pd.Series(score_avg, index=score_lang_avg.columns), ignore_index=True)
record = pd.pivot_table(
    pd.melt(all_score, id_vars=['lang'], value_vars=['consistency', 'rejection', 'knowledge'], var_name='domain', value_name=model),
    columns=['lang', 'domain']
)
res = pd.concat([record]).round(2)

print(res)