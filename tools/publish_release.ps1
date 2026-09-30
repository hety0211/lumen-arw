# Publish the built LUMEN RAW release to GitHub: commit + tag + push the source,
# then create the GitHub Release with the installer, portable ZIP, checksums and notes.
# Run with publish-release.cmd after build-release.cmd succeeded.
param([string]$Repo = 'hety0211/lumen-raw', [switch]$Draft)
$ErrorActionPreference = 'Continue'
$root = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $root
$version = ([regex]::Match((Get-Content -Raw -LiteralPath 'lumen\__init__.py'), "__version__ = '([0-9.]+)'")).Groups[1].Value
$tag = "v$version"
$publish = Join-Path $root (".publish\v" + $version.Replace('.', ''))
$packages = Join-Path $publish 'packages'
$logs = Join-Path $publish 'logs'
$log = Join-Path $logs 'publish.log'
$status = Join-Path $logs 'publish-status.txt'
New-Item -ItemType Directory -Force -Path $logs | Out-Null
Set-Content -LiteralPath $log -Value "LUMEN RAW $version GitHub publication $(Get-Date -Format s)" -Encoding UTF8

function Say([string]$text) { Write-Host $text; Add-Content -LiteralPath $log -Value $text -Encoding UTF8 }
function Step([string]$name) { Say ''; Say "==== $name  [$(Get-Date -Format HH:mm:ss)] ===="; Set-Content -LiteralPath $status -Value "running: $name" -Encoding UTF8 }
function Fail([string]$what) { Say "FAILED: $what"; Set-Content -LiteralPath $status -Value "FAILED: $what" -Encoding UTF8; throw "FAILED: $what" }
function Run([string]$what, [string]$exe, [string[]]$arguments) {
    & $exe @arguments 2>&1 | ForEach-Object { $line = "$_"; Write-Host $line; Add-Content -LiteralPath $log -Value $line -Encoding UTF8 }
    if ($LASTEXITCODE -ne 0) { Fail "$what (exit $LASTEXITCODE)" }
}
function Capture([string]$exe, [string[]]$arguments) {
    $output = & $exe @arguments 2>&1 | ForEach-Object { "$_" }
    Add-Content -LiteralPath $log -Value $output -Encoding UTF8
    return $output
}

try {
    Step 'Check built packages'
    $buildStatus = (Get-Content -Raw -LiteralPath (Join-Path $logs 'status.txt')).Trim()
    if (-not $buildStatus.StartsWith("SUCCESS $version")) { Fail "build-release has not succeeded for $version ($buildStatus)" }
    $setup = Join-Path $packages "LumenRAW-$version-Setup.exe"
    $zip = Join-Path $packages "LumenRAW-$version-Windows.zip"
    $sums = Join-Path $packages 'SHA256SUMS.txt'
    $notes = Join-Path $publish 'release-notes.md'
    foreach ($file in @($setup, $zip, $sums, $notes)) { if (-not (Test-Path -LiteralPath $file)) { Fail "missing $file" } }
    foreach ($line in Get-Content -LiteralPath $sums) {
        if (-not $line.Trim()) { continue }
        $hash, $name = $line -split '\s+', 2
        $actual = (Get-FileHash -LiteralPath (Join-Path $packages $name) -Algorithm SHA256).Hash.ToLower()
        if ($actual -ne $hash) { Fail "checksum mismatch for $name" }
        Say "checksum ok  $name"
    }

    Step 'Tools'
    $git = (Get-Command git -ErrorAction SilentlyContinue).Source
    if (-not $git) { Fail 'git is not installed' }
    $gh = (Get-Command gh -ErrorAction SilentlyContinue).Source
    if (-not $gh) { $gh = Join-Path $root '.publish\tools\gh\bin\gh.exe' }
    if (-not (Test-Path -LiteralPath $gh)) { Fail 'GitHub CLI (gh) not found' }
    Say "git: $git"
    Say "gh:  $gh"

    Step 'GitHub login'
    & $gh auth status -h github.com 2>&1 | Out-Null
    if ($LASTEXITCODE -ne 0) {
        Write-Host ''
        Write-Host 'GitHub login: a one-time code will be shown below.' -ForegroundColor Yellow
        Write-Host 'Press Enter to open https://github.com/login/device, sign in, and paste the code.' -ForegroundColor Yellow
        Write-Host ''
        & $gh auth login -h github.com -p https -w
        if ($LASTEXITCODE -ne 0) { Fail 'GitHub login was not completed' }
    }
    $account = (Capture $gh @('api', 'user', '--jq', '.login')) -join ''
    $canPush = (Capture $gh @('api', "repos/$Repo", '--jq', '.permissions.push')) -join ''
    Say "Signed in as $account; push access to ${Repo}: $canPush"
    if ($canPush -ne 'true') { Fail "account $account cannot push to $Repo" }

    Step 'Commit source'
    Run 'git fetch' $git @('fetch', 'origin', '--tags')
    $branch = ((Capture $git @('rev-parse', '--abbrev-ref', 'HEAD')) -join '').Trim()
    if ($branch -ne 'main') { Fail "expected branch main, found $branch" }
    $behind = ((Capture $git @('rev-list', '--count', 'HEAD..origin/main')) -join '').Trim()
    if ($behind -ne '0') { Fail "local main is $behind commit(s) behind origin/main; pull first" }
    Run 'git add' $git @('add', '-A')
    $staged = Capture $git @('diff', '--cached', '--name-only')
    foreach ($name in $staged) {
        if ($name -and (Test-Path -LiteralPath $name) -and (Get-Item -LiteralPath $name).Length -gt 5MB) {
            Fail "refusing to commit large file $name"
        }
    }
    Say ("staged files: " + ($staged | Where-Object { $_ }).Count)
    if (($staged | Where-Object { $_ }).Count -gt 0) {
        $message = Join-Path $logs 'commit-message.txt'
        $lines = @(
            "Release $version",
            '',
            '- Stage render cache: repair/base, tone, spatial detail and color are cached per source;',
            '  a slider only re-renders later stages. Mask alphas and the luminance reference are cached.',
            '- Local adjustments render only inside each mask''s padded bounding box (pixel-identical).',
            '- DirectML graphs for saturation/vibrance, HSL, curves, monochrome and grading; tone and',
            '  color fuse into one GPU pass without spatial tools. Graphs are generated without onnx.',
            '- Execution-provider plan in compute.provider_plan; LUMEN_COMPUTE override; CuPy opt-in.',
            '- scheduler.py: explicit WorkState conflict table and single-owner JobScheduler.',
            '- Rotating diagnostic log, single-sourced version, clean in-place installer upgrades.',
            '- 193 regression tests pass on Radeon RX 9070 XT (DirectML).',
            '',
            'Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>',
            'Claude-Session: https://claude.ai/code/session_01LKYm3S2prKHeMAYv8jFn4n'
        )
        # UTF-8 without BOM so the commit subject stays clean.
        [System.IO.File]::WriteAllLines($message, [string[]]$lines)
        Run 'git commit' $git @('commit', '-F', $message)
    }
    $head = ((Capture $git @('rev-parse', 'HEAD')) -join '').Trim()
    $existing = ((Capture $git @('rev-parse', '-q', '--verify', "refs/tags/$tag")) -join '').Trim()
    if ($existing) {
        $target = ((Capture $git @('rev-list', '-n', '1', $tag)) -join '').Trim()
        if ($target -ne $head) { Fail "tag $tag already exists on another commit ($target)" }
    } else {
        Run 'git tag' $git @('tag', '-a', $tag, '-m', "LUMEN RAW $version")
    }

    Step 'Push to GitHub'
    & $git push origin main 2>&1 | ForEach-Object { $line = "$_"; Write-Host $line; Add-Content -LiteralPath $log -Value $line -Encoding UTF8 }
    if ($LASTEXITCODE -ne 0) {
        Say 'git push failed; configuring git to use the GitHub CLI login and retrying...'
        Run 'gh auth setup-git' $gh @('auth', 'setup-git', '-h', 'github.com')
        Run 'git push main' $git @('push', 'origin', 'main')
    }
    Run 'git push tag' $git @('push', 'origin', $tag)

    Step 'GitHub Release (uploading about 1.4 GB)'
    & $gh release view $tag --repo $Repo 2>&1 | Out-Null
    if ($LASTEXITCODE -eq 0) {
        Say "Release $tag exists; replacing assets and notes."
        Run 'upload assets' $gh @('release', 'upload', $tag, $setup, $zip, $sums, '--repo', $Repo, '--clobber')
        Run 'edit release' $gh @('release', 'edit', $tag, '--repo', $Repo, '--title', "LUMEN RAW $version", '--notes-file', $notes)
    } else {
        $arguments = @('release', 'create', $tag, $setup, $zip, $sums, '--repo', $Repo, '--title', "LUMEN RAW $version",
                       '--notes-file', $notes, '--verify-tag')
        if ($Draft) { $arguments += '--draft' } else { $arguments += '--latest' }
        Run 'create release' $gh $arguments
    }

    Step 'Verify remote assets'
    $release = ((Capture $gh @('api', "repos/$Repo/releases/tags/$tag")) -join "`n") | ConvertFrom-Json
    foreach ($file in @($setup, $zip, $sums)) {
        $item = Get-Item -LiteralPath $file
        $asset = @($release.assets | Where-Object name -eq $item.Name)
        $expected = 'sha256:' + (Get-FileHash -LiteralPath $file -Algorithm SHA256).Hash.ToLower()
        if ($asset.Count -ne 1 -or $asset[0].size -ne $item.Length -or $asset[0].state -ne 'uploaded') { Fail "remote asset differs: $($item.Name)" }
        if ($asset[0].digest -and $asset[0].digest -ne $expected) { Fail "remote digest differs: $($item.Name)" }
        Say "remote ok  $($item.Name)  $($item.Length) bytes"
    }
    $report = [ordered]@{ version = $version; repository = "https://github.com/$Repo"; release = $release.html_url;
                          commit = $head; draft = $release.draft; assets = $release.assets.Count;
                          all_remote_assets_match = $true; published_at = $release.published_at }
    $report | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $publish 'published-report.json') -Encoding UTF8
    Set-Content -LiteralPath $status -Value "SUCCESS $($release.html_url)" -Encoding UTF8
    Say ''
    Say "SUCCESS: $($release.html_url)"
    exit 0
}
catch {
    Say "$_"
    if (-not ((Get-Content -Raw -LiteralPath $status) -like 'FAILED*')) { Set-Content -LiteralPath $status -Value "FAILED: $_" -Encoding UTF8 }
    exit 1
}
