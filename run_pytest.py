import subprocess, sys
result = subprocess.run(
    [sys.executable, '-m', 'pytest', 'tests', '-q', '--basetemp', '.pytest_r10', '-p', 'no:cacheprovider'],
    capture_output=True, text=True, cwd=r'D:\backup\BaoBao\PythonProgram\miaosuan', timeout=600
)
with open('pytest_result.txt', 'w', encoding='utf-8') as f:
    f.write(f'Exit code: {result.returncode}\n')
    f.write(f'Stdout length: {len(result.stdout)}\n')
    f.write(f'Stderr length: {len(result.stderr)}\n\n')
    f.write('--- STDOUT (last 2000 chars) ---\n')
    f.write(result.stdout[-2000:] if result.stdout else '(empty)')
    f.write('\n\n--- STDERR (last 2000 chars) ---\n')
    f.write(result.stderr[-2000:] if result.stderr else '(empty)')
