python revisions/combination_test/main.py --n_repeats 500 \
                                          --include_combination_tests \
                                          --n_range 200 400 600 800 1000 \
                                          --causal_effect_multiplier 0 0.5 \
                                          --main_dir 'revisions/combination_test/combination_vs_single/scm'
python revisions/combination_test/main.py --n_repeats 500 \
                                          --include_combination_tests \
                                          --n_range 200 400 600 800 1000 \
                                          --causal_effect_multiplier 0 0.5 \
                                          --scm "linear" \
                                          --covariate_pos 1 \
                                          --Z_pos 0 \
                                          --main_dir 'revisions/combination_test/combination_vs_single/linear'


python revisions/combination_test/main.py --n_repeats 500 \
                                          --n_range 200 400 600 800 1000 \
                                          --causal_effect_multiplier 0 0.5 \
                                          --exclude_default_method \
                                          --include_heuristic \
                                          --m_init 10 \
                                          --heuristic_alpha 0.1 \
                                          --include_combination_heuristic_tests \
                                          --main_dir 'revisions/combination_test/combination_vs_single/scm_heuristic'
python revisions/combination_test/main.py --n_repeats 500 \
                                          --n_range 200 400 600 800 1000 \
                                          --causal_effect_multiplier 0 0.5 \
                                          --exclude_default_method \
                                          --include_heuristic \
                                          --include_combination_heuristic_tests \
                                          --m_init 10 \
                                          --heuristic_alpha 0.1 \
                                          --scm "linear" \
                                          --covariate_pos 1\
                                          --Z_pos 0\
                                          --main_dir 'revisions/combination_test/combination_vs_single/linear_heuristic'
