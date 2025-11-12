import pandas as pd
import numpy as np

csv_noperf = pd.read_csv('benchResults.csv')
csv_perf = pd.read_csv('benchResults_perf.csv')

# 90th percentile for non-perf results
percentile_time = csv_noperf.groupby(
    ['matrix', 'mode', 'opt_level', 'schedule', 'chunk_size', 'num_threads']
)['elapsed_time'].quantile(0.90).reset_index()
percentile_time.columns = [*percentile_time.columns[:-1], 'perc90_elapsed_time']

# 90th percentile for perf results
percentile_perf = csv_perf.groupby(
    ['matrix', 'mode', 'opt_level', 'schedule', 'chunk_size', 'num_threads']
).agg({
    'elapsed_time': lambda x: x.quantile(0.90),
    'L1_miss_rate': 'mean', 
    'LLC_miss_rate': 'mean'
}).reset_index()

percentile_time.to_csv('benchResults_p90.csv', index=False)
percentile_perf.to_csv('benchResults_perf_p90.csv', index=False)

print("90th percentile saved to benchResults_p90.csv")
print("Perf analysis saved to benchResults_perf_p90.csv")