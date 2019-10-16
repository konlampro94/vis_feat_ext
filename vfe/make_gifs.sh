#! /bin/bash

# Original images
convert  -delay 50 -loop 0 tr_imgs/orig_img_*_1.png ../extras/orig1_recon.gif || exit 1;
convert  -delay 50 -loop 0 tr_imgs/orig_img_*_2.png ../extras/orig2_recon.gif || exit 1;
convert  -delay 50 -loop 0 tr_imgs/orig_img_*_3.png ../extras/orig3_recon.gif || exit 1;
convert  -delay 50 -loop 0 tr_imgs/orig_img_*_4.png ../extras/orig4_recon.gif || exit 1;
convert  -delay 50 -loop 0 tr_imgs/orig_img_*_5.png ../extras/orig5_recon.gif || exit 1;

# Out images
convert  -delay 50 -loop 0 tr_imgs/out_img_*_1.png ../extras/out1_recon.gif || exit 1;
convert  -delay 50 -loop 0 tr_imgs/out_img_*_2.png ../extras/out2_recon.gif || exit 1;
convert  -delay 50 -loop 0 tr_imgs/out_img_*_3.png ../extras/out3_recon.gif || exit 1;
convert  -delay 50 -loop 0 tr_imgs/out_img_*_4.png ../extras/out4_recon.gif || exit 1;
convert  -delay 50 -loop 0 tr_imgs/out_img_*_5.png ../extras/out5_recon.gif || exit 1;

# Diff images

convert  -delay 50 -loop 0 tr_imgs/diff_img_*_1.png ../extras/diff1_recon.gif || exit 1;
convert  -delay 50 -loop 0 tr_imgs/diff_img_*_2.png ../extras/diff2_recon.gif || exit 1;
convert  -delay 50 -loop 0 tr_imgs/diff_img_*_3.png ../extras/diff3_recon.gif || exit 1;
convert  -delay 50 -loop 0 tr_imgs/diff_img_*_4.png ../extras/diff4_recon.gif || exit 1;
convert  -delay 50 -loop 0 tr_imgs/diff_img_*_5.png ../extras/diff5_recon.gif || exit 1;


echo Success  !!!
echo ===============
echo  " $0 ended at  `date` ";
exit 0;
