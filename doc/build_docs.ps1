<#
.SYNOPSIS
    Converts Markdown files to HTML documents or Reveal.js presentations.

.DESCRIPTION
    This script uses Pandoc to convert Markdown files to:
    - HTML documents with table of contents and LaTeX/MathJax support
    - Reveal.js HTML presentations with LaTeX/MathJax support

    Requirements:
    - Pandoc must be installed and available in the system PATH
      Download from: https://pandoc.org/installing.html

.PARAMETER InputFile
    Path to the Markdown file to convert. Required.

.PARAMETER OutputFile
    Custom output file path (optional). If not specified, uses the input
    filename with .html extension in the same directory.
    Alias: -o

.PARAMETER RevealJS
    Creates a Reveal.js HTML presentation instead of a document.
    Alias: -r

.PARAMETER Theme
    Reveal.js theme for presentations. Default is "simple".
    Available themes: simple, black, white, league, beige, sky, night,
    serif, solarized, moon, dracula.
    Only used with -RevealJS.

.PARAMETER Help
    Displays this help message and exits.

.EXAMPLE
    .\build_docs.ps1 -InputFile document.md

    Converts document.md to document.html as an HTML document with TOC.

.EXAMPLE
    .\build_docs.ps1 -InputFile presentation.md -RevealJS

    Converts presentation.md to a Reveal.js HTML presentation.

.EXAMPLE
    .\build_docs.ps1 -InputFile slides.md -r -Theme night

    Creates a Reveal.js presentation with the "night" theme.

.EXAMPLE
    .\build_docs.ps1 -InputFile document.md -o output\doc.html

    Converts to HTML with a custom output path.

.NOTES
    Requires: Pandoc (https://pandoc.org/)

    The Markdown files can include:
    - YAML front matter for metadata (title, author, date)
    - LaTeX math equations (inline with $...$ or block with $$...$$)
    - Standard Markdown formatting

    Example YAML front matter:
    ---
    title: "Document Title"
    author: "Author Name"
    date: "2026-01-27"
    ---

.LINK
    https://pandoc.org/MANUAL.html

.LINK
    https://revealjs.com/
#>

[CmdletBinding()]
param(
    [Parameter(Mandatory=$false, Position=0)]
    [string]$InputFile,

    [Parameter(Mandatory=$false)]
    [Alias("o")]
    [string]$OutputFile,

    [Parameter(Mandatory=$false)]
    [Alias("r")]
    [switch]$RevealJS,

    [Parameter(Mandatory=$false)]
    [ValidateSet("simple", "black", "white", "league", "beige", "sky", "night", "serif", "solarized", "moon", "dracula")]
    [string]$Theme = "simple",

    [Parameter(Mandatory=$false)]
    [int]$Width = 960,

    [Parameter(Mandatory=$false)]
    [int]$Height = 700,

    [Parameter(Mandatory=$false)]
    [switch]$Help
)

#region Helper Functions

function Test-PandocInstalled {
    <#
    .SYNOPSIS
        Checks if Pandoc is installed and available in PATH.
    #>
    try {
        $null = Get-Command pandoc -ErrorAction Stop
        return $true
    }
    catch {
        return $false
    }
}

function Get-PandocVersion {
    <#
    .SYNOPSIS
        Retrieves the installed Pandoc version.
    #>
    try {
        $versionOutput = & pandoc --version 2>&1
        if ($versionOutput -match "pandoc\s+(\d+)\.(\d+)") {
            return @{
                Major = [int]$matches[1]
                Minor = [int]$matches[2]
                Full = "$($matches[1]).$($matches[2])"
            }
        }
    }
    catch {
        return $null
    }
}

function ConvertTo-Html {
    <#
    .SYNOPSIS
        Converts a Markdown file to HTML document or Reveal.js presentation.
    #>
    param (
        [Parameter(Mandatory=$true)]
        [string]$InputPath,

        [Parameter(Mandatory=$true)]
        [string]$OutputPath,

        [Parameter(Mandatory=$false)]
        [switch]$AsPresentation,

        [Parameter(Mandatory=$false)]
        [string]$PresentationTheme = "simple",

        [Parameter(Mandatory=$false)]
        [int]$SlideWidth = 960,

        [Parameter(Mandatory=$false)]
        [int]$SlideHeight = 700,

        [Parameter(Mandatory=$false)]
        [hashtable]$PandocVersion
    )

    # Build pandoc command arguments
    $pandocArgs = @(
        "`"$InputPath`"",
        "-o", "`"$OutputPath`"",
        "-s"  # standalone document
    )

    if ($AsPresentation) {
        # Reveal.js presentation
        # Check for local reveal.js folder for offline support
        $scriptDir = Split-Path -Parent $InputPath
        $localRevealJs = Join-Path $scriptDir "reveal.js"

        $pandocArgs += @(
            "-t", "revealjs",
            "--mathjax",
            "-V", "theme=$PresentationTheme",
            "-V", "slideNumber=true",
            "-V", "transition=none",
            "-V", "width=$SlideWidth",
            "-V", "height=$SlideHeight"
        )

        # Use local reveal.js if available for offline support
        if (Test-Path $localRevealJs) {
            $pandocArgs += @("-V", "revealjs-url=./reveal.js")
            Write-Host "Using local Reveal.js for offline support" -ForegroundColor Yellow
        }

        $formatName = "Reveal.js Presentation"
    }
    else {
        # HTML document
        $pandocArgs += @(
            "-t", "html5",
            "--mathjax",
            "--toc",
            "--toc-depth=3",
            "--highlight-style=pygments"
        )

        # For Pandoc 3.0+, use --embed-resources for self-contained output
        if ($PandocVersion -and $PandocVersion.Major -ge 3) {
            $pandocArgs += "--embed-resources"
            $pandocArgs += "--standalone"
        }

        $formatName = "HTML Document"
    }

    # Display conversion info
    Write-Host ("=" * 60) -ForegroundColor Green
    Write-Host "Markdown to $formatName Converter" -ForegroundColor Green
    Write-Host ("=" * 60) -ForegroundColor Green
    Write-Host "Input:  $InputPath" -ForegroundColor White
    Write-Host "Output: $OutputPath" -ForegroundColor White
    Write-Host "Format: $formatName" -ForegroundColor Cyan
    if ($AsPresentation) {
        Write-Host "Theme:  $PresentationTheme" -ForegroundColor Cyan
    }
    if ($PandocVersion) {
        Write-Host "Pandoc: Version $($PandocVersion.Full)" -ForegroundColor Gray
    }
    Write-Host ("-" * 60) -ForegroundColor Gray

    # Execute pandoc
    Write-Host "Converting..." -ForegroundColor Yellow
    $pandocCommand = "pandoc $($pandocArgs -join ' ')"
    Write-Verbose "Executing: $pandocCommand"

    $result = Invoke-Expression $pandocCommand 2>&1

    # Check if conversion was successful
    if ($LASTEXITCODE -eq 0 -and (Test-Path $OutputPath)) {
        $fileInfo = Get-Item $OutputPath
        Write-Host "Conversion successful!" -ForegroundColor Green
        Write-Host "  Output file: $($fileInfo.FullName)" -ForegroundColor Gray
        Write-Host "  File size: $([math]::Round($fileInfo.Length / 1KB, 2)) KB" -ForegroundColor Gray
        Write-Host ("-" * 60) -ForegroundColor Gray
        return $true
    }
    else {
        Write-Warning "Primary conversion failed. Trying simplified conversion..."

        # Fallback to simpler conversion
        $simplePandocArgs = @(
            "`"$InputPath`"",
            "-o", "`"$OutputPath`"",
            "-s",
            "--mathjax"
        )

        if ($AsPresentation) {
            $simplePandocArgs += @("-t", "revealjs")
        }

        $simplePandocCommand = "pandoc $($simplePandocArgs -join ' ')"
        Write-Verbose "Executing simplified: $simplePandocCommand"

        $result2 = Invoke-Expression $simplePandocCommand 2>&1
        if ($LASTEXITCODE -eq 0 -and (Test-Path $OutputPath)) {
            Write-Host "Simplified conversion successful!" -ForegroundColor Green
            $fileInfo = Get-Item $OutputPath
            Write-Host "  Output file: $($fileInfo.FullName)" -ForegroundColor Gray
            Write-Host "  File size: $([math]::Round($fileInfo.Length / 1KB, 2)) KB" -ForegroundColor Gray
            return $true
        }
        else {
            Write-Error "Conversion failed: $result2"
            return $false
        }
    }
}

function Show-Usage {
    <#
    .SYNOPSIS
        Displays a concise usage summary.
    #>
    Write-Host ""
    Write-Host "Markdown to HTML Converter" -ForegroundColor Cyan
    Write-Host "==========================" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "Usage: .\build_docs.ps1 -InputFile <file.md> [options]" -ForegroundColor White
    Write-Host ""
    Write-Host "Options:" -ForegroundColor Yellow
    Write-Host "  -InputFile <file>      Input Markdown file (required)"
    Write-Host "  -OutputFile, -o <file> Custom output path (optional)"
    Write-Host "  -RevealJS, -r          Create Reveal.js presentation"
    Write-Host "  -Theme <name>          Presentation theme (default: simple)"
    Write-Host "  -Help                  Display detailed help"
    Write-Host ""
    Write-Host "Output Formats:" -ForegroundColor Yellow
    Write-Host "  Default:   HTML document with table of contents"
    Write-Host "  -RevealJS: Reveal.js HTML presentation"
    Write-Host ""
    Write-Host "Examples:" -ForegroundColor Yellow
    Write-Host "  .\build_docs.ps1 -InputFile document.md"
    Write-Host "  .\build_docs.ps1 -InputFile slides.md -RevealJS"
    Write-Host "  .\build_docs.ps1 -InputFile slides.md -r -Theme night"
    Write-Host "  .\build_docs.ps1 -InputFile doc.md -o output\doc.html"
    Write-Host ""
    Write-Host "Themes for -RevealJS:" -ForegroundColor Yellow
    Write-Host "  simple, black, white, league, beige, sky, night,"
    Write-Host "  serif, solarized, moon, dracula"
    Write-Host ""
}

#endregion Helper Functions


#region Main Script

# Display help if requested
if ($Help) {
    Get-Help $PSCommandPath -Detailed
    exit 0
}

# If no input file provided, show usage
if (-not $InputFile) {
    Show-Usage
    exit 0
}

# Check if Pandoc is installed
if (-not (Test-PandocInstalled)) {
    Write-Error "Pandoc is not installed or not in PATH."
    Write-Host ""
    Write-Host "Please install Pandoc:" -ForegroundColor Yellow
    Write-Host "  Download: https://pandoc.org/installing.html" -ForegroundColor Gray
    Write-Host "  Windows:  winget install JohnMacFarlane.Pandoc" -ForegroundColor Gray
    Write-Host "  Scoop:    scoop install pandoc" -ForegroundColor Gray
    exit 1
}

# Validate input file
if (-not (Test-Path $InputFile)) {
    Write-Error "Input file does not exist: $InputFile"
    exit 1
}

if (-not ($InputFile -match '\.md$')) {
    Write-Warning "Input file does not have .md extension. Proceeding anyway..."
}

# Get Pandoc version for compatibility
$pandocVersion = Get-PandocVersion
if ($pandocVersion) {
    Write-Verbose "Detected Pandoc version: $($pandocVersion.Full)"
}

# Resolve input path
$resolvedInputPath = Resolve-Path $InputFile

# Determine output file path
if ([string]::IsNullOrWhiteSpace($OutputFile)) {
    $resolvedOutputPath = [System.IO.Path]::ChangeExtension($resolvedInputPath, ".html")
}
else {
    # Check if output directory exists, create if needed
    $outputDir = Split-Path $OutputFile -Parent
    if ($outputDir -and -not (Test-Path $outputDir)) {
        Write-Host "Creating output directory: $outputDir" -ForegroundColor Yellow
        New-Item -ItemType Directory -Path $outputDir -Force | Out-Null
    }

    # Handle relative paths
    if (-not [System.IO.Path]::IsPathRooted($OutputFile)) {
        $resolvedOutputPath = Join-Path (Get-Location) $OutputFile
    }
    else {
        $resolvedOutputPath = $OutputFile
    }

    # Ensure .html extension
    if (-not $resolvedOutputPath.EndsWith(".html")) {
        $resolvedOutputPath = [System.IO.Path]::ChangeExtension($resolvedOutputPath, ".html")
    }
}

# Perform conversion
ConvertTo-Html -InputPath $resolvedInputPath `
               -OutputPath $resolvedOutputPath `
               -AsPresentation:$RevealJS `
               -PresentationTheme $Theme `
               -SlideWidth $Width `
               -SlideHeight $Height `
               -PandocVersion $pandocVersion

#endregion Main Script
