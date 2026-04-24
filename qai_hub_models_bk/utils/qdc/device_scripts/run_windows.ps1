Set-Location C:\Temp\TestContent\
$source = "https://softwarecenter.qualcomm.com/api/download/software/sdks/Qualcomm_AI_Runtime_Community/All/{QAIRT_VERSION}/v{QAIRT_VERSION}.zip"
$output = "C:\Temp\TestContent\qairt.zip"
(New-Object System.Net.WebClient).DownloadFile($source, $output)
Expand-Archive -Path "C:\Temp\TestContent\qairt.zip" -DestinationPath "C:\Temp\TestContent\"
$env:QAIRT_HOME = "C:\Temp\TestContent\qairt\{QAIRT_VERSION}"
$env:Path = "$env:QAIRT_HOME\bin\aarch64-windows-msvc;" + $env:Path
$env:Path = "$env:QAIRT_HOME\lib\aarch64-windows-msvc;" + $env:Path
$env:ADSP_LIBRARY_PATH = "$env:QAIRT_HOME\lib\hexagon-{HEXAGON_VERSION}\unsigned"

genie-t2t-run.exe -c genie_config.json --prompt_file sample_prompt.txt | Out-File -FilePath "C:/Temp/QDC_logs/genie.log"

for ($i = 1; $i -le 10; $i++) {
    $profileName = "profile$($i).txt"
    $outputPath = "C:/Temp/QDC_logs/$profileName"
    genie-t2t-run.exe -c genie_config.json --prompt_file sample_prompt.txt --profile $outputPath
}
