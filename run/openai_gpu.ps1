$ErrorActionPreference = 'Stop'

Set-Location $PSScriptRoot

$python = 'd:/CODE/PROJETO - GLOBALVOICE/GlobalVoice-Testes/.venv/Scripts/python.exe'
$scriptPath = 'd:/CODE/PROJETO - GLOBALVOICE/GlobalVoice-Testes/whisper_benchmark_complete.py'
$projectRoot = Split-Path -Parent $scriptPath

Set-Location $projectRoot

$durations = @(
    @{
        Label = '10s'
        InputFile = Join-Path $projectRoot 'audios/10sEN.mp4'
        ReferenceFile = Join-Path $projectRoot 'textos/10sEN.txt'
    }
    @{
        Label = '30s'
        InputFile = Join-Path $projectRoot 'audios/30sEN.mp4'
        ReferenceFile = Join-Path $projectRoot 'textos/30sEN.txt'
    }
    @{
        Label = '60s'
        InputFile = Join-Path $projectRoot 'audios/60sEN.mp4'
        ReferenceFile = Join-Path $projectRoot 'textos/60sEN.txt'
    }
)

$modelSizes = @('tiny', 'base', 'small', 'medium')

$jobs = foreach ($duration in $durations) {
    foreach ($modelSize in $modelSizes) {
        @{
            DurationLabel = $duration.Label
            ModelSize = $modelSize
            InputFile = $duration.InputFile
            ReferenceFile = $duration.ReferenceFile
            OutputFile = Join-Path $projectRoot "resultados/openai-whisper/gpu/openai-$modelSize-$($duration.Label)-en-gpu.json"
        }
    }
}

if (-not (Test-Path $python)) {
    throw "Python nao encontrado em: $python"
}

if (-not (Test-Path $scriptPath)) {
    throw "Script nao encontrado: $scriptPath"
}

if (-not (Test-Path (Join-Path $projectRoot 'audios'))) {
    throw "Pasta de áudio nao encontrada em: $(Join-Path $projectRoot 'audios')"
}

if (-not (Test-Path (Join-Path $projectRoot 'textos'))) {
    throw "Pasta de textos nao encontrada em: $(Join-Path $projectRoot 'textos')"
}

$resultados = New-Object System.Collections.Generic.List[object]

foreach ($job in $jobs) {
    $outputDir = Split-Path -Parent $job.OutputFile
    if (-not (Test-Path $outputDir)) {
        New-Item -ItemType Directory -Path $outputDir -Force | Out-Null
    }

    Write-Host ''
    Write-Host ('=' * 80)
    Write-Host "Rodando openai-whisper GPU | duration=$($job.DurationLabel) | model-size=$($job.ModelSize)"
    Write-Host "Saida: $($job.OutputFile)"
    Write-Host ('=' * 80)

    $startedAt = Get-Date
    $status = 'ok'
    $errorMessage = ''

    try {
        & $python $scriptPath `
            --model openai-whisper `
            --model-size $job.ModelSize `
            --input-file $job.InputFile `
            --reference-file $job.ReferenceFile `
            --output $job.OutputFile `
            --use-gpu `
            --language en

        if ($LASTEXITCODE -ne 0) {
            throw "Processo terminou com exit code $LASTEXITCODE"
        }
    }
    catch {
        $status = 'erro'
        $errorMessage = $_.Exception.Message
        Write-Warning "Falha no teste $($job.DurationLabel)/$($job.ModelSize): $errorMessage"
    }

    $finishedAt = Get-Date
    $resultados.Add([PSCustomObject]@{
        duration = $job.DurationLabel
        model_size = $job.ModelSize
        status = $status
        started_at = $startedAt
        finished_at = $finishedAt
        output_file = $job.OutputFile
        error = $errorMessage
    }) | Out-Null

    Write-Host 'Aguardando 2 segundos antes do proximo teste...'
    Start-Sleep -Seconds 2
}

Write-Host ''
Write-Host 'Resumo final:'
$resultados | Format-Table -AutoSize

$failed = @($resultados | Where-Object { $_.status -ne 'ok' })
if ($failed.Count -gt 0) {
    Write-Error "Fila concluida com $($failed.Count) falha(s)."
    exit 1
}

Write-Host ''
Write-Host 'Fila concluida sem falhas.'
Write-Host 'Resumo final:'
$resultados | Format-Table -AutoSize
