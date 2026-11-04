@echo off
setlocal

set "SCRIPT_DIR=%~dp0"
set "DATA_ROOT=Q:\users\leejo\data\scapis\datahub"
set "OUTPUT_DIR=Q:\users\leejo\data\scapis\dicom_inventory"

if not exist "%DATA_ROOT%" (
  echo ERROR: Data folder not found: %DATA_ROOT%
  echo Check that the Q: network drive is connected.
  pause
  exit /b 2
)

python "%SCRIPT_DIR%04_build_site_dicom_inventories.py" ^
  --root "%DATA_ROOT%" ^
  --output "%OUTPUT_DIR%" ^
  --workers 8

if errorlevel 1 (
  echo.
  echo Inventory failed. Review the error above.
  pause
  exit /b 1
)

echo.
echo Inventory finished. Results are in:
echo %OUTPUT_DIR%
pause
