# Lance le rapport en LECTURE SEULE sur l'archive VPIN du serveur Shadow Run et affiche le resultat (volumes, periode, trous, distribution du VPIN).
# A lancer dans un terminal PowerShell de CE PC (invite "PS C:\...>"), jamais dans CloudShell :
#
#   powershell -ExecutionPolicy Bypass -File "C:\Users\DELL\projet API trading\scripts\shadow_report.ps1"
#
# Ne modifie rien sur le serveur et ne redemarre aucun conteneur : il fait un git pull (code du rapport), puis lance un conteneur ephemere (python:3.12-slim)
# qui monte le volume de donnees et le depot en lecture, execute workers.market_data.shadow_report, et disparait. Aucun secret n'est lu ni affiche.

param(
    [string]$ServerHost = "18.181.218.79",
    [string]$User = "ubuntu",
    [string]$Key = "$env:USERPROFILE\.ssh\nothierly_dash"
)

$ErrorActionPreference = "Stop"
$repo = "/home/ubuntu/nothierly"

if (-not (Test-Path $Key)) { Write-Host "[rapport] cle introuvable : $Key" -ForegroundColor Red; exit 1 }
$reachable = Test-NetConnection -ComputerName $ServerHost -Port 22 -WarningAction SilentlyContinue
if (-not $reachable.TcpTestSucceeded) {
    Write-Host "[rapport] port 22 injoignable : ton IP a change ? Mets a jour la regle SSH du groupe de securite (source = ton IP /32)." -ForegroundColor Red
    exit 1
}

function Invoke-Remote([string]$Label, [string]$Command) {
    Write-Host ""
    Write-Host "== $Label" -ForegroundColor Cyan
    & ssh -o StrictHostKeyChecking=accept-new -i $Key "$User@$ServerHost" $Command
    if ($LASTEXITCODE -ne 0) { Write-Host "[rapport] ECHEC a l'etape '$Label' (code $LASTEXITCODE). Colle-moi les lignes ci-dessus." -ForegroundColor Red; exit $LASTEXITCODE }
}

Invoke-Remote "1/2 git pull" "git -C $repo pull origin red-team-audit"
$run = "docker run --rm -v nothierly_shadow_data:/data -v ${repo}:/src:ro -w /src -e PYTHONDONTWRITEBYTECODE=1 -e NOTHIERLY_DATA_DIR=/data python:3.12-slim python -m workers.market_data.shadow_report --db /data/vpin_history.sqlite3"
Invoke-Remote "2/2 rapport VPIN (lecture seule)" $run
Write-Host ""
Write-Host "[rapport] termine. Copie-moi tout ce qui est affiche entre '== 2/2' et cette ligne." -ForegroundColor Green
