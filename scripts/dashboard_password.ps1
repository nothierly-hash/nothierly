# Copie le mot de passe du tableau de bord du serveur dans le presse-papiers SANS l'afficher, puis l'efface du presse-papiers apres 60 secondes.
# A lancer dans un terminal PowerShell de CE PC (invite "PS C:\...>"), jamais dans CloudShell :
#
#   powershell -ExecutionPolicy Bypass -File .\scripts\dashboard_password.ps1
#
# Puis dans le tableau de bord : utilisateur "admin", champ mot de passe : Ctrl+V. Le mot de passe n'est ecrit nulle part sur ce PC et n'est pas affiche.

param(
    [string]$ServerHost = "18.181.218.79",
    [string]$User = "ubuntu",
    [string]$Key = "$env:USERPROFILE\.ssh\nothierly_dash",
    [int]$ClearAfterSeconds = 60
)

$ErrorActionPreference = "Stop"

if (-not (Test-Path $Key)) { Write-Host "[mdp] cle introuvable : $Key" -ForegroundColor Red; exit 1 }
$reachable = Test-NetConnection -ComputerName $ServerHost -Port 22 -WarningAction SilentlyContinue
if (-not $reachable.TcpTestSucceeded) {
    Write-Host "[mdp] port 22 injoignable : ton IP a change ? Mets a jour la regle SSH du groupe de securite (source = ton IP /32)." -ForegroundColor Red
    exit 1
}

$line = & ssh -o StrictHostKeyChecking=accept-new -i $Key "$User@$ServerHost" "grep '^DASHBOARD_PASSWORD=' /home/ubuntu/nothierly/.env"
if ($LASTEXITCODE -ne 0 -or -not $line) { Write-Host "[mdp] lecture impossible (code $LASTEXITCODE)." -ForegroundColor Red; exit 1 }

$prefix = "DASHBOARD_PASSWORD="
$password = ([string]$line).Trim()
if (-not $password.StartsWith($prefix)) { Write-Host "[mdp] reponse inattendue du serveur." -ForegroundColor Red; exit 1 }
$password = $password.Substring($prefix.Length)
if ($password.Length -lt 8) { Write-Host "[mdp] mot de passe vide ou trop court sur le serveur : lance scripts/setup_shadow_env.sh --complete." -ForegroundColor Red; exit 1 }

Set-Clipboard -Value $password
$password = $null
$line = $null
Write-Host "[mdp] copie dans le presse-papiers (rien n'est affiche). Utilisateur : admin. Colle avec Ctrl+V." -ForegroundColor Green
Write-Host "[mdp] le presse-papiers sera efface dans $ClearAfterSeconds s. Ne ferme pas cette fenetre avant."
Start-Sleep -Seconds $ClearAfterSeconds
Set-Clipboard -Value " "
Write-Host "[mdp] presse-papiers efface." -ForegroundColor Yellow
