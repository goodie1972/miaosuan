with open('pytest_out.txt', 'r', encoding='utf-8', errors='replace') as f:
    content = f.read()
lines = content.splitlines()
with open('pytest_summary.txt', 'w', encoding='utf-8') as out:
    out.write(f'Total lines: {len(lines)}\n')
    # Find the summary line
    for i, line in enumerate(lines):
        if 'passed' in line or 'failed' in line or 'error' in line.lower():
            out.write(f'Line {i}: {line}\n')
    out.write('\n--- Last 15 lines ---\n')
    for line in lines[-15:]:
        out.write(line + '\n')
