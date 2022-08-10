import argparse
from multiprocessing import cpu_count

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def get_args():
    argparser = argparse.ArgumentParser()

    # General arguments
    argparser.add_argument('--n_cpu', type=int, default=cpu_count()-2)
    argparser.add_argument('--seed', type=int, default=None)
    argparser.add_argument('--n_repeats', type=int, default=1000)
    argparser.add_argument('--main_dir', type=str, default="revisions/combination_test")
    argparser.add_argument('--exclude_default_method', type=str2bool, nargs='?', const=True, default=False)

    # Resampling and combination parameters
    argparser.add_argument('--n_combinations', type=int, default=100)
    argparser.add_argument('--m_rate', type=float, default=0.5)
    argparser.add_argument('--replacement', type=str, default='False')
    argparser.add_argument('--use_heuristic_for_combination', type=str2bool, nargs='?', const=True, default=False)
    
    # Simulation parameters
    argparser.add_argument('--effect_vanish_power', type=float, default=0)
    argparser.add_argument('--causal_effect_multiplier', type=float, nargs='+', default=[0])
    argparser.add_argument('--n_range', type=int, nargs='+', default=[int(10**p) for p in [2, 2.5, 3, 3.5, 4]])
    argparser.add_argument('--use_linear_scm', type=str2bool, nargs='?', const=True, default=False)
    
    # Test parameters
    argparser.add_argument('--alpha', type=float, default=0.05)
    argparser.add_argument('--test_conf_int', type=str2bool, nargs='?', const=True, default=False)
    argparser.add_argument('--response_pos', type=int, default=2)
    argparser.add_argument('--covariate_pos', type=int, default=0)
    argparser.add_argument('--Z_pos', type=int, default=1)
    argparser.add_argument('--reweight_target', type=int, default=1)
    argparser.add_argument('--reweight_covariate', type=int, nargs='+', default=[0])
    argparser.add_argument('--target_var_reduction_factor', type=float, default=1)
    

    # Heuristic parameters
    argparser.add_argument('--exclude_heuristic', type=str2bool, nargs='?', const=True, default=False)
    argparser.add_argument('--quantile_repeats', type=int, default=10)
    argparser.add_argument('--m_init', type=int, default=None)
    
    # CPT parameters
    argparser.add_argument('--exclude_cpt', type=str2bool, nargs='?', const=True, default=False)
    argparser.add_argument('--CPT_M', type=int, default=300)
    argparser.add_argument('--CPT_n_step', type=int, default=40)

    return argparser.parse_args()