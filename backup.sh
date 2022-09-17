#/bin/sh
mkdir -p bp/Docker bp/models/student bp/models/teacher bp/losses bp/src bp/utils bp/results
cp losses/*.py bp/losses/
cp src/* bp/src/
cp Docker/* bp/Docker/
cp utils/*.py bp/utils/
cp models/student/*.py bp/models/student/
cp models/teacher/*.py bp/models/teacher/
cp results/*.png bp/results/
cp *.py *.sh *.jpg bp/
