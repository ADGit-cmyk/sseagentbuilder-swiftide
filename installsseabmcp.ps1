param(
    [switch]$DryRun
)

# === CONFIG ===
$BaseDir      = $PSScriptRoot
$ZipName      = "swiftide-0.32.1.zip"
$ZipPath      = Join-Path $BaseDir $ZipName
$ExtractedDir = Join-Path $BaseDir "swiftide-0.32.1"
$PatchRoot    = Join-Path $BaseDir "swiftide-files"
$SwiftIDEURL  = "https://github.com/bosun-ai/swiftide/archive/refs/tags/v0.32.1.zip"

function Ensure-Dir { param($p) if (-not (Test-Path $p)) { New-Item -ItemType Directory -Path $p -Force | Out-Null } }

function Remove-ReparseIfPresent {
    param([string]$path)
    if (Test-Path $path) {
        try {
            $it = Get-Item -LiteralPath $path -Force
            if ($it.Attributes -band [System.IO.FileAttributes]::ReparsePoint) {
                if (-not $DryRun) { Remove-Item -LiteralPath $path -Force -ErrorAction Stop }
            }
        } catch { }
    }
}

function Copy-File-Force {
    param([string]$src, [string]$dst)
    $dstDir = Split-Path -Parent $dst
    Ensure-Dir $dstDir
    Remove-ReparseIfPresent -path $dst

    if (-not $DryRun) {
        Copy-Item -LiteralPath $src -Destination $dst -Force -ErrorAction Stop
    }
}

# === SCRIPT START ===
Set-Location $BaseDir

# 1) Download zip if missing
if (-not (Test-Path $ZipPath)) {
    Invoke-WebRequest -Uri $SwiftIDEURL -OutFile $ZipPath -UseBasicParsing -ErrorAction Stop
}

# 2) Extract zip if needed
if (-not (Test-Path $ExtractedDir)) {
    Expand-Archive -Path $ZipPath -DestinationPath $BaseDir -Force -ErrorAction Stop
}

# 3) Validate patch root exists
if (-not (Test-Path $PatchRoot)) { exit 1 }

# 4) Copy all patch files into extracted folder
$patchResolved  = (Resolve-Path $PatchRoot).ProviderPath
$targetResolved = (Resolve-Path $ExtractedDir).ProviderPath

$files = Get-ChildItem -Path $patchResolved -Recurse -File -Force
foreach ($f in $files) {
    $rel  = $f.FullName.Substring($patchResolved.Length).TrimStart('\','/')
    $dest = Join-Path $targetResolved $rel
    Copy-File-Force -src $f.FullName -dst $dest
}
