# FusionPHI
FusionPHI is a tool for phage-host interactions prediction
## Required Dependencies
  | environment | version   |
  | ----------- | --------- |
  | Python      | 3.8 |
  | pytorch     | 2.4.1 |
  | numpy       | 1.24.3 |
  | sklearn     | 1.3.0 |
  | biopython | 1.78 |
  | matplotlib | 3.7.2 |
  | pandas | 2.0.3 |

You also need to download [MEME](https://meme-suite.org/meme/meme-software/5.5.5/meme-5.5.5.tar.gz) toolkit to get motif sequences.
After downloading, use the following command to decompress this toolkit:
```
tar zxf meme-5.5.5.tar.gz
```
Then，use
```
cd meme-5.5.5
./configure --prefix=$pwd/meme/meme_install --with-url=http://meme-suite.org/ --enable-build-libxml2 --enable-build-libxslt
```
You should use “pwd” command to find your position, fill in the $pwd field in the above instruction.
```
make
make test
make install
vim ~/.bashrc
export PATH=$pwd/meme/meme_install/bin:$PATH 
```
Enter the above command and validate your setup by:
```
meme --help
```
Suppose you want to extract the dna motif of NZ_CP025774.fasta. An example is as follows:
```
meme host_genes/NZ_CP025774.fasta -dna -time 18000 -mod anr -nmotifs 5 -minw 5 -maxw 72 -evt 0.01 -o ./Single_Results/NZ_CP025774_result
```
We sugguset you use the command below to extract the motif sequence of all fasta files in the folder at one time
```
_使用biopython识别VHM1426基因组基序.py
```
## Usage

## Contact
Please contact Li(2542479361@qq.com or GitHub Issues) with any questions, concerns or comments.

Thank you for using FusionPHI!
