#/bin/sh
mkdir -p bp/Docker bp/student bp/teacher bp/losses bp/src bp/utils bp/results
cp losses/*.py bp/losses/
cp src/* bp/src/
cp Docker/* bp/Docker/
cp utils/*.py bp/utils/
cp student/*.py bp/student/
cp teacher/*.py bp/teacher/
cp results/*.png bp/results/
cp *.py *.sh *.jpg bp/
