#!/bin/bash
cd ../Super-Resolution
python3 ./main.py ./../PSNR/HALF.png
cd ../PSNR
rm *intermediate_.*
python3 psnr.py
rm ./*\(2x\).*
cd ..

