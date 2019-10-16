#! /bin/bash

# generate data => {ext_frames?} needed to run training
./prep_video.sh 1  || exit 1;
./prep_video.sh 2  || exit 1;
./prep_video.sh 3  || exit 1;
./prep_video.sh 4  || exit 1;
./prep_video.sh 5  || exit 1;

# run models
python ae.py --view 1 --images_dir ext_frames1 || exit 1;
python ae.py --view 2 --images_dir ext_frames2 || exit 1;
python ae.py --view 3 --images_dir ext_frames3 || exit 1;
python ae.py --view 4 --images_dir ext_frames4 || exit 1;
python ae.py --view 5 --images_dir ext_frames5 || exit 1;


echo Success  !!!
echo ===============
echo  " $0 ended at  `date` ";
exit 0;
