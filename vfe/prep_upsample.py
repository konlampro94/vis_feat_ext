import os
import sys
import csv

FPS=100

def main():

    if len(sys.argv) > 1:
        view = sys.argv[1]
        print(f"view is:\t {view}")
        mode = sys.argv[2]
        print(f"mode is:\t {mode}")
    else:
        print("sys argv error! please try again.This script is called with two arguments")
        print("E.g python prep_upsample.py 1 train")
        sys.exit(1)

    folder = mode + view
    print(f"folder is:\t {folder}")    
    if not os.path.exists(folder):
        os.makedirs(folder)
    

    if int(view) not in [1,2,3,4,5]:
        print("Error with view")
        sys.exit(1)

    if mode == "train":

        path_list = ["video/"+str(i)+"/"+view for i in range(1,42)]
        csv_file="align_"+view+".csv"
        csv_reader = csv.reader(open(csv_file, newline='\n'), delimiter=',')
        names = []
        vals = []

        for row in csv_reader:
    
            names.append(row[0])
            vals.append(row[1])
  
        for path in path_list:
            cur_files=os.listdir(path)
            for file in cur_files:

                file_mp4 = file.split(".")[0]
                val = vals[names.index(file_mp4)]
                tmp_folder=folder+"/"+file_mp4
                #print(tmp_folder)
                os.makedirs(tmp_folder)
                os.system(f"ffmpeg -i {path}/{file} -vframes {val} -vf fps=fps={FPS} {tmp_folder}/{file_mp4}%03d.png")

    elif mode == "test":

        path_list = ["video/"+str(i)+"/"+view for i in range(42,54)]
        csv_file="align_"+view+"_test.csv"
        csv_reader = csv.reader(open(csv_file, newline='\n'), delimiter=',')
        names = []
        vals = []
        
        for row in csv_reader:
            
            names.append(row[0])
            vals.append(row[1])
  
        for path in path_list:
            cur_files=os.listdir(path)
            for file in cur_files:

                file_mp4 = file.split(".")[0]
                val = vals[names.index(file_mp4)]
                tmp_folder=folder+"/"+file_mp4
                #print(tmp_folder)
                os.makedirs(tmp_folder)
                os.system(f"ffmpeg -i {path}/{file} -vframes {val}  -vf fps=fps={FPS} {tmp_folder}/{file_mp4}%03d.png")
    
    else:
        print("Error with mode")
        sys.exit(1)



if __name__ == "__main__":
    main()