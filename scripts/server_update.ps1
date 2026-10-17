# Met a jour le serveur Shadow Run depuis ce PC Windows : git pull, rebuild/restart des conteneurs, etat des conteneurs.
# A lancer dans un terminal PowerShell de CE PC (invite "PS C:\...>"), jamais dans CloudShell :
#
#   powershell -ExecutionPolicy Bypass -File .\scripts\server_update.ps1
#
# Utilise la cle SSH et l'adresse ci-dessous ; n'affiche et ne lit aucun secret (pas de mot de passe, pas de .env). Pour changer les services a
# reconstruire : -Services "nothierly-agents nothierly-dashboard".

param(
    [string]$ServerHost = "18.181.218.79",
    [string]$User = "ubuntu",
    [string]$Key = "$env:USERPROFILE\.ssh\nothierly_dash",
    [string]$Services = "nothierly-agents nothierly-dashboard"
)

$ErrorActionPreference = "Stop"
$repo = "/home/ubuntu/nothierly"
$compose = "docker compose -f $repo/docker-compose.yml -f $repo/docker-compose.platform.yml -f $repo/docker-compose.shadow.yml --project-directory $repo"

if (-not (Test-Path $Key)) { Write-Host "[maj] cle introuvable : $Key" -ForegroundColor Red; exit 1 }
$reachable = Test-NetConnection -ComputerName $ServerHost -Port 22 -WarningAction SilentlyContinue
if (-not $reachable.TcpTestSucceeded) {
    Write-Host "[maj] port 22 injoignable : ton IP a change ? Mets a jour la regle SSH du groupe de securite (source = ton IP /32)." -ForegroundColor Red
    exit 1
}

function Invoke-Remote([string]$Label, [string]$Command) {
    Write-Host ""
    Write-Host "== $Label" -ForegroundColor Cyan
    & ssh -o StrictHostKeyChecking=accept-new -i $Key "$User@$ServerHost" $Command
    if ($LASTEXITCODE -ne 0) { Write-Host "[maj] ECHEC a l'etape '$Label' (code $LASTEXITCODE). Colle-moi les lignes ci-dessus." -ForegroundColor Red; exit $LASTEXITCODE }
}

Invoke-Remote "1/3 git pull" "git -C $repo pull origin red-team-audit"
Invoke-Remote "2/3 build + redemarrage ($Services) : plusieurs minutes, sans affichage au debut" "$compose up -d --build $Services"
Invoke-Remote "3/3 etat des conteneurs" "docker ps --format 'table {{.Names}}\t{{.Status}}'"
Write-Host ""
Write-Host "[maj] termine." -ForegroundColor Green
