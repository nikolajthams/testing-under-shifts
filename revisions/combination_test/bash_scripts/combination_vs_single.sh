python revisions/combination_test/main.py --n_repeats 500 \
                                          --n_range 200 400 600 800 1000 \
                                          --causal_effect_multiplier 0 0.5\
                                          --exclude_heuristic \
                                          --exclude_cpt \
                                          --main_dir 'revisions/combination_test/combination_vs_single/scm'
python revisions/combination_test/main.py --n_repeats 500 \
                                          --n_range 200 400 600 800 1000 \
                                          --causal_effect_multiplier 0 0.5\
                                          --exclude_heuristic \
                                          --exclude_cpt \
                                          --scm "linear" \
                                          --covariate_pos 1\
                                          --Z_pos 0\
                                          --main_dir 'revisions/combination_test/combination_vs_single/linear'
python revisions/combination_test/main.py --n_repeats 500\
                                          --n_range 200 400 600 800 1000 \
                                          --causal_effect_multiplier 0 0.5\
                                          --exclude_heuristic \
                                          --exclude_cpt \
                                          --scm "paper_scm" \
                                          --main_dir 'revisions/combination_test/combination_vs_single/paper_scm'
