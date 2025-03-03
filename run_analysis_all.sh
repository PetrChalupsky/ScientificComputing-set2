#!/bin/bash

echo "Running full analysis including long runs"
echo "Running calculate_gray_scott"
python scripts/calculate_gray_scott.py
echo "Running calculate_monte_carlo"
python scripts/calculate_monte_carlo.py
echo "Running calculate_DLA_Laplace"
python scripts/calculate_DLA_Laplace.py
echo "Running calculate_speedup_DLA"
python scripts/calculate_speedup_DLA.py
echo "Running calculate_find_optimal_omega"
python scripts/calculate_find_optimal_omega.py

echo "Creating plots"
echo "Running visualize_gray_scott"
python scripts/visualize_gray_scott.py
echo "Running visualize_monte_carlo"
python scripts/visualize_monte_carlo.py
echo "Running visualize_DLA_Laplace"
python scripts/visualize_DLA_Laplace.py
echo "Running visualize_speedup_DLA"
python scripts/visualize_speedup_DLA.py
echo "Running visualize_optimal_omega"
python scripts/visualize_optimal_omega.py

echo "Analysis finished, hooman!"

