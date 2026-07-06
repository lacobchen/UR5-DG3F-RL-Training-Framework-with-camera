@echo off
setlocal EnableExtensions

call conda activate isaaclab

set "PROJECT_ROOT=D:\UR5_Project\UR5_RL_with_Camera"
set "ISAACLAB_ROOT=C:\Users\lamal\IsaacLab"
set "ISAACSIM_PATH=%ISAACLAB_ROOT%\_isaac_sim"
set "EXP_PATH=%ISAACSIM_PATH%\apps"
set "ISAAC_PATH=%ISAACSIM_PATH%"
set "CARB_APP_PATH=%ISAACSIM_PATH%\kit"


call "%ISAACSIM_PATH%\setup_python_env.bat"

set "PATH=%ISAACSIM_PATH%;%ISAACSIM_PATH%\kit;%ISAACSIM_PATH%\kit\plugins;%ISAACSIM_PATH%\kit\kernel;%ISAACSIM_PATH%\kit\kernel\plugins;%ISAACSIM_PATH%\kit\python;%ISAACSIM_PATH%\kit\python\DLLs;%PATH%"

set "PYTHONPATH=%PROJECT_ROOT%;%ISAACSIM_PATH%\kit\kernel\py;%ISAACSIM_PATH%\python_packages;%ISAACSIM_PATH%\exts;%ISAACSIM_PATH%\extscache;%ISAACSIM_PATH%\kit\python\Lib\site-packages;%PYTHONPATH%"

cd /d "%PROJECT_ROOT%"

echo.
echo [INFO] PROJECT_ROOT  = %PROJECT_ROOT%
echo [INFO] ISAACLAB_ROOT = %ISAACLAB_ROOT%
echo [INFO] ISAACSIM_PATH = %ISAACSIM_PATH%
echo [INFO] CARB_APP_PATH = %CARB_APP_PATH%

echo.
echo [INFO] Start official Isaac Lab training...

call "%ISAACLAB_ROOT%\isaaclab.bat" -p "%ISAACLAB_ROOT%\scripts\reinforcement_learning\rsl_rl\train.py" ^
  --task UR5-PickCube-v0 ^
  --num_envs 2 ^
  --max_iterations 2 ^
  --headless ^
  --enable_cameras

echo.
echo [INFO] isaaclab.bat returned errorlevel = %ERRORLEVEL%

echo.
echo [INFO] Finished command. If traceback appeared above, training failed even if errorlevel is 0.
pause
exit /b %ERRORLEVEL%