# Everything chapter 5 still needs, in one sequential pass. Sequential on purpose: two
# renderers on one card measure the card's contention, and a scorer running beside a
# render starves the submission thread.
$ErrorActionPreference = "Continue"
$repo = "C:\Users\macad\Documents\_Projects\BambooEngine"
$log  = $args[0]
function Step($name, $arguments) {
    "=== $name : $(Get-Date -Format o)" | Out-File -Append -Encoding utf8 $log
    & python @arguments 2>&1 | Out-File -Append -Encoding utf8 $log
    "=== $name exit=$LASTEXITCODE" | Out-File -Append -Encoding utf8 $log
}
Set-Location $repo

Step "diffmap (8.5.1)"        @("tools/campaign.py","diffmap")
Step "reference-length (E5)"  @("tools/campaign.py","reference-length")
Step "breakdown (8.4)"        @("tools/campaign.py","--only","K1","breakdown")
Step "memory (8.6, P5)"       @("tools/campaign.py","--only","K1","memory")
Step "temporal (K4, 8.5)"     @("tools/campaign.py","--only","K1","temporal")
Step "temporal-report"        @("tools/campaign.py","--only","K1","temporal-report")
Step "levers (Q13)"           @("tools/campaign.py","--only","K1","levers")
Step "importance (blok 7)"    @("tools/recon.py","importance")
Step "importance-report"      @("tools/recon.py","importance-report")
Step "direct (blok 8)"        @("tools/recon.py","direct")
Step "direct-report"          @("tools/recon.py","direct-report")

"=== QUEUE COMPLETE : $(Get-Date -Format o)" | Out-File -Append -Encoding utf8 $log
