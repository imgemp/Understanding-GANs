# Understanding-GANs

## Example Usage
source activate ugan (ugans is old)
python ugans/run.py $(cat examples/args/circles/con/00.txt)
or on gypsum
srun -p m40-long --gres=gpu:1 examples/run.sh "examples/args/celebA/con/00.txt" "-latdim 1 -lat_dis_reg 0.0"
where first quotes contain path to an arguments file and second quotes contain argument modifications (if desired) to use instead of what is in the file

cd to the results folder of interest and run the following to see tensorboard in the browser (navigate to http://localhost:6006)
tensorboard --logdir=./
