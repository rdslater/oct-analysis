import pandas as pd
from pydicom import dcmread
from pathlib import Path

data = pd.read_csv("Located_files.csv")
DATA_PATH = "/mnt/scratch/group/domalpally/data/Score2/"
SUBJ = []
SESS = []
ACQ = []
File = []
SESS_LAT = []
Modality = []
Laterality = []
Shape = []
count = 0
for _, row in data.iterrows():
    #build path
    loc = Path(DATA_PATH) / "SUBJECTS" / row['RandomizedSubjectId'] / "SESSIONS" / row['TimePoint'] / "ACQUISITIONS" / row['Procedure'] / "FILES"
    #list files
    file_list = list(loc.glob("*.*"))
    #loop through
    for file in file_list:
        if str(file)[-3:]=="dcm":
            dcm = dcmread(file)
            shape = dcm.pixel_array.shape
            laterality = dcm.ImageLaterality
            modality = dcm.Modality
        else:
            modality = "NA"
            laterality = "NA"
            shape = "NA"
        SUBJ.append(row['RandomizedSubjectId'])
        SESS.append(row['TimePoint'])
        ACQ.append(row['Procedure'])
        File.append(str(file).split("/")[-1])
        SESS_LAT.append(row['Laterality'])
        Modality.append(modality)
        Laterality.append(laterality)
        Shape.append(shape)
        count+=1

output = pd.DataFrame({"Subject":SUBJ,"Session":SESS,"Acquisition":ACQ,"FileName":File,"ListedLaterality":SESS_LAT,"Modality":Modality,"FileLaterality":Laterality,"Shape":Shape})
output.to_csv("file_list.csv")
