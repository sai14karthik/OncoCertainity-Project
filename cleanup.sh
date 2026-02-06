#!/bin/bash
# Cleanup script to remove log files, cache, and temporary files
# Usage: ./cleanup.sh

# Remove all log files
rm -f output_*.log error_*.log *.log

# Remove Python cache
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null
find . -type f -name "*.pyc" -delete 2>/dev/null

# Remove .egg-info
find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null

# Remove temporary files
rm -f submit_*_tmp.sh *.tmp *.temp builder.py 2>/dev/null

# Remove test results
rm -rf test_results/ 2>/dev/null

echo "Cleanup complete!"
