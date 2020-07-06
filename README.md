# DeepPhy
Deep Learning toolbox for inferring phylogenetic trees. Makes use of PhyDL and Seg-Gen.

# Getting Started!
This is a python 3.8.3 program. It is recommended that you install the latest version of python.

There are a few critical dependencies...
## PyTorch
To install PyTorch, go to `pytorch.org` and select the correct specs for your machine. 
This program does utilize the gpu, so you will need to have a NVidea gpu that supports cuda. Make sure to select a cuda version for installation.

## Seq-Gen
We use Seq-Gen to generate our DNA sequence data. Head over to www.github.com/rambaut/Seq-Gen to install/clone. 
This program is not exactly Windows friendly. The workaround: install the Ubuntu command line from the Microsoft Store! 
Then follow all the steps in the Seq-Gen repo. If you have a Mac OS or Linux machine, simply follow their directions.

For windows, after you run the make file through Ubuntu:

run,
`cd /mnt/c/Seq-Gen`

This will navigate you to the Seq-Gen directory. Now you can run Seq-Gen commands to generate data! Or, you can take advantage of the code in data_gen.py.
My code will generate a batch file that will create multiple (or just one, you choose) datasets and commands at once. Once you run data_gen, simply type into Ubuntu,
`./mysh.sh`
and the data will be saved to a number of folders in the Seq-Gen directory. I've also commented areas in my code that are places to customize and change how your data is generated and saved.



