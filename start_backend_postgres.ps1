# Starts the backend on PostgreSQL. Run from the project root:
#   powershell -ExecutionPolicy Bypass -File .\start_backend_postgres.ps1
$secure = Read-Host "PostgreSQL password for user 'postgres'" -AsSecureString
$plain  = [System.Net.NetworkCredential]::new("", $secure).Password
$env:DATABASE_URL = "postgresql+psycopg://postgres:$([uri]::EscapeDataString($plain))@localhost:5432/churn"
$plain = $null
Set-Location backend
Write-Host "Starting the backend on PostgreSQL (database: churn)..." -ForegroundColor Green
.\venv\Scripts\python.exe wsgi.py



#cd ..
# @'
# # Starts the backend on PostgreSQL (database: churn). Works from any folder.
# $root = $PSScriptRoot
# $secure = Read-Host "PostgreSQL password for user 'postgres'" -AsSecureString
# $plain  = [System.Net.NetworkCredential]::new("", $secure).Password
# $env:DATABASE_URL = "postgresql+psycopg://postgres:$([uri]::EscapeDataString($plain))@localhost:5432/churn"
# $plain = $null
# Set-Location (Join-Path $root "backend")
# Write-Host "Starting the backend on PostgreSQL (database: churn)..." -ForegroundColor Green
# & (Join-Path $root "backend\venv\Scripts\python.exe") wsgi.py
# '@ | Set-Content -Encoding utf8 start_backend_postgres.ps1
# powershell -ExecutionPolicy Bypass -File .\start_backend_postgres.ps1