#!/bin/bash

echo "Running analysis without long runs"
echo "Running calculate_gray_scott"
python scripts/calculate_gray_scott.py
echo "Running calculate_monte_carlo"
python scripts/calculate_monte_carlo.py
echo "Running calculate_DLA_Laplace"
python scripts/calculate_DLA_Laplace.py

echo "Creating plots"
echo "Running visualize_gray_scott"
python scripts/visualize_gray_scott.py
echo "Running visualize_monte_carlo"
python scripts/visualize_monte_carlo.py
echo "Running visualize_DLA_Laplace"
python scripts/visualize_DLA_Laplace.py

echo "Analysis finished, hooman!"

