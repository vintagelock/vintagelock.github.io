<#
.SYNOPSIS
    Strips UTF-8 BOMs and replaces Unicode punctuation with ASCII equivalents.

.DESCRIPTION
    Walks a file or folder (recursively) and replaces common Unicode characters
    that cause problems in source control or plain-text toolchains:

        U+2014  em dash        ->  -
        U+2013  en dash        ->  -
        U+2192  right arrow    ->  ->
        U+2190  left arrow     ->  <-
        U+00D7  multiplication ->  x
        U+2019  right quote    ->  '
        U+2018  left quote     ->  '
        U+201C  left dbl quote ->  "
        U+201D  right dbl quote -> "
        U+2500  box drawings     ->  -
        U+2502  box drawings     ->  |
        U+2026  ellipsis          ->  ...
        U+25BC  black down-pointing triangle -> v
    Also removes UTF-8 BOMs (EF BB BF) if present.

.PARAMETER Path
    Path to a single file, or a directory to process recursively.

.PARAMETER Extensions
    File extensions to process when Path is a directory.
    Defaults to: .cpp .h .c .cc .cxx .hpp .txt .cmake .md

.PARAMETER DryRun
    Report what would change without writing any files.

.EXAMPLE
    .\Fix-Encoding.ps1 -Path . -DryRun
    Show all files that would be modified.

.EXAMPLE
    .\Fix-Encoding.ps1 -Path .\src
    Fix all source files under .\src in-place.

.EXAMPLE
    .\Fix-Encoding.ps1 -Path .\platform\win32\win32_ipc.cpp
    Fix a single file.
#>

[CmdletBinding()]
param(
    [Parameter(Mandatory, Position = 0)]
    [string] $Path,

    [string[]] $Extensions = @('.cpp', '.h', '.c', '.cc', '.cxx', '.hpp', '.txt', '.cmake', '.md'),

    [switch] $DryRun
)

$replacements = [ordered]@{
    [char]0x2014 = '-'    # em dash
    [char]0x2013 = '-'    # en dash
    [char]0x2192 = '->'   # right arrow
    [char]0x2190 = '<-'   # left arrow
    [char]0x00D7 = 'x'    # multiplication sign
    [char]0x2019 = "'"    # right single quote
    [char]0x2018 = "'"    # left single quote
    [char]0x201C = '"'    # left double quote
    [char]0x201D = '"'    # right double quote
    [char]0x2500 = '-'    # box drawings light horizontal
    [char]0x2502 = '|'    # box drawings light vertical
    [char]0x2026 = '...'  # ellipsis
    [char]0x25BC = 'v'    # black down-pointing triangle
}

$utf8NoBom = New-Object System.Text.UTF8Encoding($false)

function Get-DisplayChar($ch) {
    "U+{0:X4} ({1})" -f ([int][char]$ch), $ch
}

function Repair-File($filePath) {
    $bytes = [System.IO.File]::ReadAllBytes($filePath)

    $hasBom = ($bytes.Length -ge 3 -and $bytes[0] -eq 0xEF -and $bytes[1] -eq 0xBB -and $bytes[2] -eq 0xBF)
    $contentBytes = if ($hasBom) { $bytes[3..($bytes.Length - 1)] } else { $bytes }

    $original = [System.Text.Encoding]::UTF8.GetString($contentBytes)
    $modified = $original

    $issues = [System.Collections.Generic.List[string]]::new()

    if ($hasBom) {
        $issues.Add("  BOM present")
    }

    foreach ($kvp in $replacements.GetEnumerator()) {
        $unicodeChar = $kvp.Key
        $asciiReplacement = $kvp.Value

        if ($modified.Contains($unicodeChar)) {
            $count = ($modified.ToCharArray() | Where-Object { $_ -eq $unicodeChar }).Count
            $issues.Add("  $(Get-DisplayChar $unicodeChar) -> '$asciiReplacement'  ($count occurrence$(if ($count -ne 1) {'s'}))")
            $modified = $modified.Replace([string]$unicodeChar, $asciiReplacement)
        }
    }

    # Flag any remaining non-ASCII characters not covered by the replacement table
    $unknownNonAscii = $modified.ToCharArray() | Where-Object { [int]$_ -gt 127 } | Sort-Object | Get-Unique
    foreach ($ch in $unknownNonAscii) {
        $issues.Add("  WARNING: unknown non-ASCII $(Get-DisplayChar $ch) - not replaced")
    }

    if ($issues.Count -eq 0) {
        return
    }

    $relativePath = $filePath.Replace((Resolve-Path $Path).Path, '').TrimStart('\').TrimStart('/')
    if ($DryRun) {
        Write-Host "$relativePath" -ForegroundColor Cyan
        foreach ($issue in $issues) {
            Write-Host $issue -ForegroundColor Yellow
        }
    }
    else {
        [System.IO.File]::WriteAllText($filePath, $modified, $utf8NoBom)
        Write-Host "$relativePath" -ForegroundColor Green
        foreach ($issue in $issues) {
            Write-Host $issue -ForegroundColor DarkGray
        }
    }
}

$resolvedPath = Resolve-Path $Path -ErrorAction Stop

if (Test-Path $resolvedPath -PathType Leaf) {
    if ($DryRun) {
        Write-Host "[DRY RUN] Checking: $resolvedPath" -ForegroundColor Magenta
    }
    Repair-File $resolvedPath.Path
}
elseif (Test-Path $resolvedPath -PathType Container) {
    # Directory - recurse
    $include = $Extensions | ForEach-Object { "*$_" }
    $files = Get-ChildItem -Recurse -Path $resolvedPath -Include $include

    if ($DryRun) {
        Write-Host "[DRY RUN] Scanning $($files.Count) files under $resolvedPath" -ForegroundColor Magenta
        Write-Host ""
    }

    foreach ($file in $files) {
        Repair-File $file.FullName
    }

    if ($DryRun) {
        Write-Host ""
        Write-Host "[DRY RUN] Done. Run without -DryRun to apply changes." -ForegroundColor Magenta
    }
    else {
        Write-Host ""
        Write-Host "Done." -ForegroundColor Green
    }
}
else {
    Write-Error "Path not found: $Path"
    exit 1
}