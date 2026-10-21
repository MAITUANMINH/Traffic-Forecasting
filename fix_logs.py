# Sửa log bị lỗi tiếng Việt do PowerShell ghi bằng bảng mã cp437
import glob

for path in glob.glob('logs/*.txt'):
    with open(path, encoding='utf-8', errors='replace') as f:
        lines = f.read().splitlines()
    fixed, changed = [], 0
    for line in lines:
        try:
            new = line.encode('cp437').decode('utf-8')
            if new != line:
                changed += 1
            fixed.append(new)
        except (UnicodeEncodeError, UnicodeDecodeError):
            fixed.append(line)          # dòng đã đúng hoặc không sửa được thì giữ nguyên
    if changed:
        with open(path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(fixed) + '\n')
    print(f'{path}: sửa {changed} dòng')