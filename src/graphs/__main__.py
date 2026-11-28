"""Gera todas as figuras do relatório em results/images/:  python -m src.graphs"""
from src.graphs import age_mae_by_group, identity_far_frr

for module in (age_mae_by_group, identity_far_frr):
    module.main()
