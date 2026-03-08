
rm -f output_*.log error_*.log *.log

find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null
find . -type f -name "*.pyc" -delete 2>/dev/null

find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null

rm -f submit_*_tmp.sh *.tmp *.temp builder.py 2>/dev/null

rm -rf test_results/ 2>/dev/null

echo "Cleanup complete!"
