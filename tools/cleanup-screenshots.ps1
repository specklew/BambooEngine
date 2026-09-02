# Frees the screenshot tree of every measurement directory that no longer feeds a chapter.
# Keeps: the reference sets, the archived 2026-08-31 campaign (numbers only, PNGs stripped),
# its FLIP maps, and the live campaign output. Every report .md was copied to
# docs/measurement-archive/ before this script was written.
$root = "C:\Users\macad\Documents\_Projects\BambooEngine\Raytracer\SavedUserData\Screenshots"
$keep = @("campaign-refs","campaign-maps-prefix-2026-08-31","campaign-prefix-2026-08-31",
          "recon-refs","recon-curve-refs","eval-refs","eval-refs-lowexp","unbiased-ref")
$before = (Get-PSDrive C).Free

# 1. the contaminated partial re-run (disk filled mid-flight, captures truncated)
if (Test-Path "$root\campaign") { Remove-Item "$root\campaign" -Recurse -Force }

# 2. every ad-hoc / historical directory
$del = Get-ChildItem $root -Directory | Where-Object { ($keep -notcontains $_.Name) -and ($_.Name -notlike "ref*") }
Write-Output "deleting $($del.Count) directories"
foreach ($d in $del) { Remove-Item $d.FullName -Recurse -Force -ErrorAction SilentlyContinue }

# 3. the archived campaign keeps its numbers (.json/.log/.md) but not its pixels
Get-ChildItem "$root\campaign-prefix-2026-08-31" -Recurse -File -Filter *.png | Remove-Item -Force

$after = (Get-PSDrive C).Free
Write-Output ("freed {0:N1} GB - free now {1:N1} GB" -f (($after-$before)/1GB), ($after/1GB))
Write-Output "remaining: $((Get-ChildItem $root -Directory).Count) directories"
