# Sửa log bị lỗi tiếng Việt (bảng mã cp437) và đổi mọi log về UTF-8
import glob

def read_text(path):
    raw = open(path, 'rb').read()
    if raw.startswith(b'\xff\xfe') or raw.startswith(b'\xfe\xff'):
        return raw.decode('utf-16'), 'UTF-16'
    if raw.startswith(b'\xef\xbb\xbf'):
        return raw.decode('utf-8-sig'), 'UTF-8 BOM'
    try:
        return raw.decode('utf-8'), 'UTF-8'
    except UnicodeDecodeError:
        return raw.decode('cp1252', errors='replace'), 'khác'

for path in sorted(glob.glob('logs/*.txt')):
    text, enc = read_text(path)
    fixed, changed = [], 0
    for line in text.splitlines():
        try:
            new = line.encode('cp437').decode('utf-8')
            if new != line:
                changed += 1
            line = new
        except (UnicodeEncodeError, UnicodeDecodeError):
            pass                      # dòng đã đúng thì giữ nguyên
        fixed.append(line)
    with open(path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(fixed) + '\n')
    print(f'{path}: đọc dạng {enc}, sửa {changed} dòng')