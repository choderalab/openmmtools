#!/bin/bash

git checkout master > /dev/null
python trajectory_io_benchmarks.py benchmark_master

echo ""
echo "****"
echo ""
git checkout multistate-xtc-traj > /dev/null
python trajectory_io_benchmarks.py benchmark_multistate-xtc-traj