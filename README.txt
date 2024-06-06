The .yml file is the conda environment I use. If you make a new conda environment with:
conda env create --name env_name --file=environment.yml
you should have all the packages you need to run the code. The models.py file contains models, and the Helpers.py file contains assorted helper functions for data manipulation and sending output to wandb.

The multi_dict.ipynb file is close to a minimal training code example. It should run with no changes. 

To track runs with wanbd, change the 'Track_run' flag to true, and enter your own Wandb api key. After every training epoch you can store any values you want to keep track of into a dictionary, dict, and call  wandb.log(dict) to log the values in wanbd. 

If you have any questions or something isn't working, just shoot me an email.
