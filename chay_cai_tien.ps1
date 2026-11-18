# Chạy 3 cấu hình cải tiến (mỗi cấu hình ~1 giờ). Bản gốc B_BigST đã có từ giữa kỳ.
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$runs = @(
  @("--local_graph", "--name", "C_A"),
  @("--predict_delta", "--name", "C_B"),
  @("--local_graph", "--predict_delta", "--name", "C_AB")
)
foreach ($a in $runs) {
  $ten = $a[-1]
  python -X utf8 -u run_fast.py --use_long --epochs 50 @a 2>&1 | Tee-Object -FilePath "logs/$ten.txt"
}