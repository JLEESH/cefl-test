# cefl-test
CEFL (in development).

Training only the embeddings may be difficult.

Compare

``gollem_tests.py --mode train_to_zero --freezing-mode freeze_for_fl``

with

``gollem_tests.py --mode train_to_zero --freezing-mode freeze_for_pe_cft``

cf. ``files/gollem_tests_train_to_zeros_freezing_mode_comp_``