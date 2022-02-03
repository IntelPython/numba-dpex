# Original code: https://github.com/IntelPython/dpctl/blob/0e595728eb9dfc943774b654035e9b339bde8dce/.github/workflows/conda-package.yml#L220-L250
echo "OCL_ICD_FILENAMES=C:\Miniconda\Library\lib\intelocl64.dll" | Out-File -FilePath $env:GITHUB_ENV -Encoding utf8 -Append
try {$list = Get-Item -Path HKLM:\SOFTWARE\Khronos\OpenCL\Vendors | Select-Object -ExpandProperty Property } catch {$list=@()}
if ($list.count -eq 0) {
    if (-not (Test-Path -Path HKLM:\SOFTWARE\Khronos)) {
        New-Item -Path HKLM:\SOFTWARE\Khronos
    }
    if (-not (Test-Path -Path HKLM:\SOFTWARE\Khronos\OpenCL)) {
        New-Item -Path HKLM:\SOFTWARE\Khronos\OpenCL
    }
    if (-not (Test-Path -Path HKLM:\SOFTWARE\Khronos\OpenCL\Vendors)) {
        New-Item -Path HKLM:\SOFTWARE\Khronos\OpenCL\Vendors
    }
    New-ItemProperty -Path HKLM:\SOFTWARE\Khronos\OpenCL\Vendors -Name C:\Miniconda\Library\lib\intelocl64.dll -Value 0
    try {$list = Get-Item -Path HKLM:\SOFTWARE\Khronos\OpenCL\Vendors | Select-Object -ExpandProperty Property } catch {$list=@()}
    Write-Output $(Get-Item -Path HKLM:\SOFTWARE\Khronos\OpenCL\Vendors)
    # Now copy OpenCL.dll into system folder
    $system_ocl_icd_loader="C:\Windows\System32\OpenCL.dll"
    $python_ocl_icd_loader="C:\Miniconda\Library\bin\OpenCL.dll"
    Copy-Item -Path $python_ocl_icd_loader -Destination $system_ocl_icd_loader
    if (Test-Path -Path $system_ocl_icd_loader) {
        Write-Output "$system_ocl_icd_loader has been copied"
        $acl = Get-Acl $system_ocl_icd_loader
        Write-Output $acl
    } else {
        Write-Output "OCL-ICD-Loader was not copied"
    }
    # Variable assisting OpenCL CPU driver to find TBB DLLs which are not located where it expects them by default
    echo "TBB_DLL_PATH=C:\Miniconda\Library\bin" | Out-File -FilePath $env:GITHUB_ENV -Encoding utf8 -Append
}
