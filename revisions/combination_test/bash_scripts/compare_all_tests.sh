python revisions/combination_test/main.py --n_repeats 1000\
                                          --causal_effect_multiplier 0 0.3 0.6\
                                          --m_init 10
                                          --main_dir 'revisions/combination_test/compare_all_tests/non_linear'
python revisions/combination_test/main.py --n_repeats 1000 \
                                          --causal_effect_multiplier 0 0.1 \
                                          --m_init 10 \
                                          --use_linear_scm \
                                          --covariate_pos 1\
                                          --Z_pos 0\
                                          --main_dir 'revisions/combination_test/compare_all_tests/linear'