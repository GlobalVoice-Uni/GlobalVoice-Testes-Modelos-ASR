param(
    [ValidateSet("tiny", "base", "small", "medium", "large")]
    [string]$model = "base",

    [ValidateSet("cpu", "gpu")]
    [string]$device = "cpu",

    [ValidateSet("en", "pt-br")]
    [string]$language = "en",

    [int]$duration = 20,
    [int]$context = 5
)

$pythonExe = ".\.venv\Scripts\python.exe"
if (-not (Test-Path $pythonExe)) {
    Write-Host "Python da venv não encontrado em .venv\Scripts\python.exe" -ForegroundColor Red
    exit 1
}

Write-Host "Iniciando realtime benchmark..." -ForegroundColor Cyan
Write-Host "Modelo: $model | Device: $device | Idioma: $language | Duracao: ${duration}s | Contexto: $context" -ForegroundColor Gray

$argsList = @(
    "realtime_tests/realtime_benchmark.py",
    "--model", $model,
    "--device", $device,
    "--language", $language,
    "--duration", "$duration",
    "--context", "$context"
)

& $pythonExe @argsList
