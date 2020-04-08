#!/bin/sh

git clone https://github.com/tanreinama/OCR_Japanease-models.git
cd OCR_Japanease-models
./concatenate.sh
mv *.model ../models
cd ..
rm -rf OCR_Japanease-models
