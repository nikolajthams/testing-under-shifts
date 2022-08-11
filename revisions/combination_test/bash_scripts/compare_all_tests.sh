python revisions/combination_test/main.py --n_repeats 1000\
                                          --causal_effect_multiplier 0 0.3 0.6\
                                          --include_competing_tests \
                                          --include_resample_hsic \
                                          --m_init 10
                                          --main_dir 'revisions/combination_test/compare_all_tests/non_linear'
python revisions/combination_test/main.py --n_repeats 1000 \
                                          --causal_effect_multiplier 0 0.1 \
                                          --include_competing_tests \
                                          --include_resample_hsic \
                                          --m_init 10 \
                                          --scm "linear" \
                                          --covariate_pos 1\
                                          --Z_pos 0\
                                          --main_dir 'revisions/combination_test/compare_all_tests/linear'
python revisions/combination_test/main.py --n_repeats 1000\
                                          --causal_effect_multiplier 0 0.3 0.6\
                                          --include_competing_tests \
                                          --include_resample_hsic \
                                          --m_init 10
                                          --scm "paper_scm" \
                                          --main_dir 'revisions/combination_test/compare_all_tests/paper_scm'
