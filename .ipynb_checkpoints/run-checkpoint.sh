#!/bin/bash
python ./Face-Recognition/Reused\ Saved\ Model.py
cd Super-Resolution
python3 ./main.py ./../Images/*/*
cd ../Images
rm ./*/*intermediate_.*
cd ..
python ./Face-Recognition/Reused\ Saved\ Model.py
cd Images
rm ./*/*\(2x\).*
