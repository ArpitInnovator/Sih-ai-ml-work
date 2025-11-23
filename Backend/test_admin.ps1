# Quick test script for admin endpoints
Write-Host "Testing Admin Endpoints..." -ForegroundColor Green
Write-Host ""

# Test root admin endpoint
Write-Host "Testing /api/v1/admin/ ..." -ForegroundColor Cyan
try {
    $response = Invoke-RestMethod -Uri "http://localhost:8000/api/v1/admin/" -Method Get
    Write-Host "✓ Admin endpoint is working!" -ForegroundColor Green
    Write-Host "Response:" -ForegroundColor Yellow
    $response | ConvertTo-Json -Depth 3
} catch {
    Write-Host "✗ Error: $_" -ForegroundColor Red
}

Write-Host ""
Write-Host "API Documentation: http://localhost:8000/docs" -ForegroundColor Cyan

