#! /bin/bash
#--spec_model vd_epoch_20.pth --models_dir out --view 1 --mode train

# prep_upsample.sh --view  --mode
#./prep_upsample.sh 1 train || exit 1;
#./prep_upsample.sh 1 test || exit 1;
#./prep_upsample.sh 2 train || exit 1;
#./prep_upsample.sh 2 test || exit 1;
#./prep_upsample.sh 3 train || exit 1;
#./prep_upsample.sh 3 test || exit 1;
#./prep_upsample.sh 4 train || exit 1;
#./prep_upsample.sh 4 test || exit 1;
#./prep_upsample.sh 5 train || exit 1;
#./prep_upsample.sh 5 test || exit 1;


python prep_upsample.py 1 train || exit 1;
python prep_upsample.py 1 test || exit 1;
python prep_upsample.py 2 train || exit 1;
python prep_upsample.py 2 test || exit 1;
python prep_upsample.py 3 train || exit 1;
python prep_upsample.py 3 test || exit 1;
python prep_upsample.py 4 train || exit 1;
python prep_upsample.py 4 test || exit 1;
python prep_upsample.py 5 train || exit 1;
python prep_upsample.py 5 test || exit 1;



python extract.py --spec_model vd_epoch_20.pth --models_dir out1 --view 1 --mode train || exit 1;
python extract.py --spec_model vd_epoch_20.pth --models_dir out1 --view 1 --mode test || exit 1;
python extract.py --spec_model vd_epoch_20.pth --models_dir out2 --view 2 --mode train || exit 1;
python extract.py --spec_model vd_epoch_20.pth --models_dir out2 --view 2 --mode test || exit 1;
python extract.py --spec_model vd_epoch_20.pth --models_dir out3 --view 3 --mode train || exit 1;
python extract.py --spec_model vd_epoch_20.pth --models_dir out3 --view 3 --mode test || exit 1;
python extract.py --spec_model vd_epoch_20.pth --models_dir out4 --view 4 --mode train || exit 1;
python extract.py --spec_model vd_epoch_20.pth --models_dir out4 --view 4 --mode test || exit 1;
python extract.py --spec_model vd_epoch_20.pth --models_dir out5 --view 5 --mode train || exit 1;
python extract.py --spec_model vd_epoch_20.pth --models_dir out5 --view 5 --mode test || exit 1;

# Generate scp files
cat ./train_video_1/*.scp > train_video_1_3.scp || exit 1;
cat ./test_video_1/*.scp > test_video_1_3.scp   || exit 1;
cat ./train_video_2/*.scp > train_video_2_3.scp  || exit 1;
cat ./test_video_2/*.scp > test_video_2_3.scp     || exit 1;
cat ./train_video_3/*.scp > train_video_3_3.scp   || exit 1;
cat ./test_video_3/*.scp > test_video_3_3.scp     || exit 1;
cat ./train_video_4/*.scp > train_video_4_3.scp    || exit 1;
cat ./test_video_4/*.scp > test_video_4_3.scp       || exit 1;
cat ./train_video_5/*.scp > train_video_5_3.scp     || exit 1;
cat ./test_video_5/*.scp > test_video_5_3.scp       || exit 1;

rm ./train_video_1/*.scp 
rm ./test_video_1/*.scp 
rm ./train_video_2/*.scp 
rm ./test_video_2/*.scp 
rm ./train_video_3/*.scp 
rm ./test_video_3/*.scp 
rm ./train_video_4/*.scp
rm ./test_video_4/*.scp 
rm ./train_video_5/*.scp 
rm ./test_video_5/*.scp 


cp  *.scp ../../../kaldi/egs/oulu_kaldi_exps/ || exit 1;


echo Success  !!!
echo ===============
echo  " $0 ended at  `date` ";
exit 0;
