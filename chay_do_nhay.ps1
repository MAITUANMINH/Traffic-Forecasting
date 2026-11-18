# Thi nghiem do nhay r va tau (6 lan chay, ~6 gio)
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$runs = @(
  @("--tau", "0.2", "--name", "S_tau0.2"),
  @("--tau", "0.6", "--name", "S_tau0.6"),
  @("--tau", "1.0", "--name", "S_tau1.0"),
  @("--tau", "0.2", "--random_feature_dim", "16", "--name", "S_r16"),
  @("--tau", "0.2", "--random_feature_dim", "32", "--name", "S_r32"),
  @("--random_feature_dim", "16", "--local_graph", "--predict_delta", "--name", "C_AB_r16")
)
foreach ($a in $runs) {
  $ten = $a[-1]
  python -X utf8 -u run_fast.py --use_long --epochs 50 @a 2>&1 | Tee-Object -FilePath "logs/$ten.txt"
}
