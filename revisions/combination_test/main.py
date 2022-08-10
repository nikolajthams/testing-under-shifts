import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
import numpy as np
import pandas as pd
import json
from multiprocessing import Pool, cpu_count
from functools import partial
from tqdm import tqdm
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

    # Save args file
    with open(os.path.join(MAIN_DIR, "args", f"{time_string}.json"), "w") as f:
        json.dump(vars(args), f, indent=2)