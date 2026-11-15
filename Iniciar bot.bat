@echo off
rem Dois cliques para ligar o bot: ele reconhece o monstro na tela (ou pede F9 em cima dele).
rem Para parar: Ctrl+C nesta janela (o mouse num canto da tela tambem para, mas so
rem enquanto o bot mexe o mouse).
rem Antes confere o Python (3.10 ou mais novo) e instala os pacotes que faltarem.
rem Gravado em ASCII com CRLF: o cmd precisa do CRLF para os goto/:rotulo.
cd /d "%~dp0"
python -c "import sys; sys.exit(sys.version_info < (3, 10))" >nul 2>&1 || goto sem_python
python -c "import cv2, numpy, mss, pyautogui, pygetwindow, keyboard" >nul 2>&1 || goto instalar
:ligar
python main.py %*
goto fim

:instalar
echo Instalando os pacotes do bot (so na primeira vez; precisa de internet)...
python -m pip install -r requirements.txt || goto sem_pacotes
goto ligar

:sem_pacotes
echo.
echo Nao consegui instalar os pacotes. Veja o erro acima e tente de novo.
goto fim

:sem_python
echo Nao achei o Python 3.10 ou mais novo.
echo Instale em https://www.python.org/downloads/ e marque "Add python.exe to PATH".
echo Depois, dois cliques aqui de novo.

:fim
pause
