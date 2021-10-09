import sys
import subprocess
from compare_stats import compare_stats

# Set user defined arguments
version1 = str(sys.argv[1])
version2 = str(sys.argv[2])
runs = int(sys.argv[3])

# Define agent paths
agent1_path = f'submission_{version1}/main.py'
agent2_path = f'submission_{version2}/main.py'

# Execute battles
for i in range(runs):
    subprocess.check_call(f'lux-ai-2021 {agent1_path} {agent2_path}', shell=True)
    print('\n')

# Generate and save stats files
compare_stats()
