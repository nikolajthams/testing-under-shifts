python revisions/combination_test/main.py --n_repeats 100 \
                                          --n_range 200 400 600 800 1000 \
                                          --causal_effect_multiplier 0 0.1 0.2 \
                                          --exclude_heuristic \
                                          --scm "linear" \
                                          --covariate_pos 1\
                                          --Z_pos 0\
                                          --main_dir 'revisions/combination_test/cpt_vs_pt'