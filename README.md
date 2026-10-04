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

# Sept 2026
Initial runs with batch size of 4-8 and 256x256x96 yielded 67% accuracy on a single question (Cystoid Spaces).  Doing some reaserch OCTs are not an even pixel shape with the Z (slice dimension) often being "bigger" (1 pixel movement in Z is roughly 5 in X and Y).  Thus the voxels are "skinny" in X and Y and "Long" in 7.

Base on a research suggestion, going to 256x256x128 to try and keey the Z dimension from being flattened did improve results to ~75% accuracy.  During this time it was noted that the GPU is data starved.  I am attacking that in a number of ways.  I recently made numpy arrays in with dedicated uint8 encoding.  I am also considering saving a smaller format (such as the 256x256x128) to save read time.

