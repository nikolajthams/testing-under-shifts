import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
import numpy as np
import pandas as pd
import json
from multiprocessing import Pool
from functools import partial
from tqdm import tqdm
from statsmodels.stats.proportion import proportion_confint
from datetime import datetime
from revisions.combination_test.arguments import get_args
from revisions.combination_test.experiment import experiment



if __name__ == "__main__":
    args = get_args()
    time_string = datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
    MAIN_DIR = args.main_dir


    # Define partial experiment function which takes args as input
    experiment_partial = partial(experiment, args=args)

    # Multiprocess
    pool = Pool(args.n_cpu)
    res = np.array(
        list(tqdm(pool.imap_unordered(experiment_partial, range(args.n_repeats)), total=args.n_repeats)))
    pool.close()
    res = [item for sublist in res for item in sublist]
        
    # Check if relevant folders exists
    for folder in ["results", "args"]:
        if not os.path.exists(os.path.join(MAIN_DIR, folder)):
            os.makedirs(os.path.join(MAIN_DIR, folder))
    
    # Save data
    df = pd.DataFrame(res)
    
    df.to_csv(os.path.join(MAIN_DIR, "results", f"{time_string}.csv"), index=False)
    df.to_csv(os.path.join(MAIN_DIR, "latest.csv"), index=False)
    
    def get_ci(rejects):
        lower, upper = proportion_confint(count=rejects.sum(), nobs = len(rejects), alpha=0.05, method="wilson")
        return lower, rejects.mean(), upper

    df = df.groupby([i for i in df.columns if i != "reject"]).aggregate(get_ci)
    for n,col in enumerate(["lower", "mean", "upper"]):
        df[col] = df['reject'].apply(lambda x: x[n])
    df = df.drop('reject',axis=1).rename({"mean": "reject"}, axis=1).reset_index()

    
    df.to_csv(os.path.join(MAIN_DIR, "results", f"{time_string}.csv"), index=False)
    df.to_csv(os.path.join(MAIN_DIR, "latest.csv"), index=False)


    # Save args file
    with open(os.path.join(MAIN_DIR, "args", f"{time_string}.json"), "w") as f:
        json.dump(vars(args), f, indent=2)
