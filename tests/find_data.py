import pandas as pd
from pathlib import Path
import os
from pydicom import dcmread

DATA_PATH = "/mnt/scratch/group/domalpally/data/Score2"

def main():
    count = 0
    found_files = []
    df = pd.read_csv("../local_data/octanalysis_input.csv")
    for index, row in df.iterrows():
        loc = Path(DATA_PATH) / "SUBJECTS" / row['RandomizedSubjectId'] / "SESSIONS" / row['TimePoint'] / "ACQUISITIONS" / row['Procedure'] / "FILES"
        file_list = list(loc.glob("*.dcm"))
        if len(file_list) == 1:
            dcm = dcmread(file_list[0], stop_before_pixels=True)
            if row["Laterality"]==dcm.ImageLaterality:
                found_files.append(file_list[0])
            else:
                found_files.append("Single File, Wrong Laterality")
                print(row["Laterality"],dcm.ImageLaterality)
        else:
            find_xml = list(loc.glob("*.xml"))
            if len(find_xml)==1:
                target = find_xml[0]
                try:
                    dcm = dcmread(str(find_xml[0]).replace("LayersFile.xml","dcm.dcm"), stop_before_pixels=True)
                    if row['Laterality'] == dcm.ImageLaterality:
                        found_files.append(file_list[0])
                    else:
                        found_files.append("Multiple Files, Wrong Laterality")
                except:
                    found_files.append("Could not read file or file not foundi matching xml")
            else:
                found_files.append("Multiple XML files")

        count+=1
        if count % 200 == 0:
            print(count)
    df["File_Loc"] = found_files
    df.to_csv("Located_files.csv")
    

if __name__=="__main__":
    main()
