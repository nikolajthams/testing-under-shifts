import numpy as np
from itertools import product
from resample_and_test import ShiftTester
from cpt import CPT, Model_GAM
from revisions.combination_test.data_generation import scm, linear_scm, paper_scm
from revisions.combination_test.competing_tests import get_GCM, get_KCI, get_pcit, get_fcit
from revisions.combination_test.weights_and_tests import get_permutation_p_val, get_permutation_T, get_weight_func, get_p_val_func, get_T, get_HSIC

def experiment(seed=None, args=None):
    out = []

    for n, ce_multiplier in product(args.n_range, args.causal_effect_multiplier):
        causal_effect = ce_multiplier*(n**args.effect_vanish_power)
        T = get_T(causal_effect=causal_effect, alpha=args.alpha, test_conf_int=args.test_conf_int, response_pos=args.response_pos, covariate_pos=args.covariate_pos)
        p_val_func = get_p_val_func(response_pos=args.response_pos, covariate_pos=args.covariate_pos)
        weight_func = get_weight_func(linear=(args.scm == "linear"), reweight_target=args.reweight_target, reweight_covariate=args.reweight_covariate, target_var_reduction_factor=args.target_var_reduction_factor)
        psi = ShiftTester(weight_func, T=T, replacement=args.replacement, p_val=p_val_func)

        # Set resampling rate (when heuristic is not in use)
        m = int(n**args.m_rate)
        
        # Simulate data
        if args.scm == "linear":
            data = linear_scm(n, causal_effect=causal_effect, seed=seed)
        elif args.scm == "paper_scm":
            data = paper_scm(n, causal_effect=causal_effect, seed=seed)
        else:
            data = scm(n, causal_effect=causal_effect, seed=seed)
        

        # Conduct single test
        if not args.exclude_default_method:
            out.append({
                    "method": "single-test",
                    "reject": psi.test(data, m=m),
                    "n": n,
                    "causal_effect": causal_effect
                })
            # Conduct combination tests
            for combination_type in ["hartung", "meinshausen", "cct"]:
                if args.use_heuristic_for_combination:
                    p_cutoff = np.quantile(np.random.uniform(size=(1000, args.quantile_repeats)).min(axis=1), 0.05)
                    m_combination = psi.tune_m(data, j_x=[0], j_y=[1], gaussian=True, repeats=args.quantile_repeats, cond=[0], m_init=args.m_init, p_cutoff=p_cutoff, m_factor=1.3)
                else: 
                    m_combination = m
                out.append({
                        "method": combination_type,
                        "reject": psi.combination_test(data, m=m_combination, n_combinations=args.n_combinations, method=combination_type, warn=False, alpha=args.alpha),
                        "n": n,
                        "causal_effect": causal_effect
                    })
        if args.include_resample_hsic:
            T_HSIC = get_HSIC(args, return_p_val=False)
            pval_HSIC = get_HSIC(args, return_p_val=True)
            psi_HSIC = ShiftTester(weight_func, T=T_HSIC, replacement=args.replacement, p_val=pval_HSIC)
            out.append({
                    "method": "single-test-HSIC",
                    "reject": psi_HSIC.test(data, m=m),
                    "n": n,
                    "causal_effect": causal_effect
                })
            # Conduct combination tests
            for combination_type in ["hartung", "meinshausen", "cct"]:
                if args.use_heuristic_for_combination:
                    p_cutoff = np.quantile(np.random.uniform(size=(1000, args.quantile_repeats)).min(axis=1), 0.05)
                    m_combination = psi_HSIC.tune_m(data, j_x=[0], j_y=[1], gaussian=True, repeats=args.quantile_repeats, cond=[0], m_init=args.m_init, p_cutoff=p_cutoff, m_factor=1.3)
                else: 
                    m_combination = m
                out.append({
                        "method": f"{combination_type}-HSIC",
                        "reject": psi_HSIC.combination_test(data, m=m_combination, n_combinations=args.n_combinations, method=combination_type, warn=False, alpha=args.alpha),
                        "n": n,
                        "causal_effect": causal_effect
                    })
        # Conduct single test with heuristic
        if not args.exclude_heuristic:
            p_cutoff = np.quantile(np.random.uniform(size=(1000, args.quantile_repeats)).min(axis=1), 0.05)
            m_heuristic = psi.tune_m(data, j_x=[0], j_y=[1], gaussian=True, repeats=args.quantile_repeats, cond=[0], m_init=args.m_init, p_cutoff=p_cutoff, m_factor=1.3)
            out.append({
                    "method": "heuristic",
                    "reject": psi.test(data, m=m_heuristic),
                    "n": n,
                    "causal_effect": causal_effect
                })
        # Conduct CPT test
        if not args.exclude_cpt:
            cpt = CPT(model=Model_GAM(), M=args.CPT_M, n_step=args.CPT_n_step)
            p_val = cpt.get_p_val(X=data[:,args.covariate_pos], Y = data[:,args.response_pos], Z = data[:,args.Z_pos])
            out.append({
                    "method": "cpt",
                    "reject": 1.0*(p_val < args.alpha),
                    "n": n,
                    "causal_effect": causal_effect
                })
            
            # Use resampling and then a permutation test for comparison with the CPT
            perm_p_val_func = get_permutation_p_val(n_permutations=args.CPT_M, response_pos=args.response_pos, covariate_pos=args.covariate_pos)
            perm_T = get_permutation_T(n_permutations=args.CPT_M, response_pos=args.response_pos, covariate_pos=args.covariate_pos, alpha=args.alpha)
            psi_permutation_test = ShiftTester(weight_func, p_val=perm_p_val_func, T=perm_T, replacement=args.replacement)
            
            # Single permutation test
            out.append({
                "method": "resample+permutation",
                "reject": psi_permutation_test.test(data, m=m),
                "n": n,
                "causal_effect": causal_effect
            })
            
            # Conduct combination tests with permutation test for independence
            out.append({
                    "method": "resample+permutation+combination",
                    "reject": psi_permutation_test.combination_test(data, m=m, n_combinations=args.n_combinations, method="hartung", warn=False, alpha=args.alpha),
                    "n": n,
                    "causal_effect": causal_effect
                })

        if args.include_competing_tests:
            pv_gcm = get_GCM(args, return_p_val=False)
            pv_kci = get_KCI(args, return_p_val=False)
            pv_pcit = get_pcit(args, return_p_val=False)
            pv_fcit = get_fcit(args, return_p_val=False)
            for method, test in zip(["GCM", "KCI", "pcit", "fcit"], [pv_gcm, pv_kci, pv_pcit, pv_fcit]):
                out.append({
                        "method": method,
                        "reject": test(data),
                        "n": n,
                        "causal_effect": causal_effect
                    })

    return out
