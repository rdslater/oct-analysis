## DATA
Data Setup
Initial data was supplied by amitha is is stored in file <insert file here>

I had to create a pivot table of all the questions and that file is stored in putthenamehere.py

Once we had the data in the proper format, we needed to find the dicoms from the M drive (uploaded to flywheel in Score2) downladed here to pR.  The script that inventoried every file and looked at modality and laterlity.  I filtered to have files that only matched the listed lateriality AND were OPT modality.  In addition, looked at the dims of the dimension and any OPT that had less than 49 slices was excluded.  I also excluded the 200,1000,200 OCT because it just did not look standard.

check_data.py runs a basic transform and load to batch every file and check before we kick off.

The final list is contained in octanalysis_input.csv

From there the data is split into 5 folds by subject ID.  Fold 0 will be validation to start and Folds != 0 will be the training set.

# Recreate
conda env create -f environment.yml -n my_new_env

