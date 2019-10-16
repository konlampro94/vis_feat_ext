#! /bin/bash



 
if [ ! -d "video"  ] ; then
	echo "Linking cropped_mouth_mp4_phrase here"
	read -p "Please give me the relative path to the video folder" rel_path
	: "${rel_path:=../../OuluVS2/cropped_mouth_mp4_phrase}"
	ln -s $rel_path video ||  exit 1;
	
fi

echo "Sleeping for 5"
sleep 5


echo "Argument checking..."

if [[ "$#" -ge 1 ]] ; then 
	view=$1
else
	read -p "Please give me view(1-5) to extract: " view
fi
#read -p "Please give me view(1-5) to extract: " view
folder="ext_frames$view" || exit 1;
echo $folder
sleep 5




if [ ! -d "$folder" ] ; then

	mkdir $folder || exit 1;

	#for i in {1..53}  ; do
	#	mkdir ext_frames/$i || exit 1;
	#done

	files=$(ls video)
	#echo $files
	file_front_mp4=$(for file in $files ; do  ls video/$file/1 ;  done )
	#echo $file_front_mp4


	ffmpeg_inst=`which ffmpeg | grep "not found"`
	if [ ! -z "$ffmpef_inst" ] ; then
		# Assuming Debian distro
		echo "Trying to install ffmpeg package!!"
		#sudo apt install ffmpeg -y || exit 1;
		# Assuming Centos
		#sudo yum install ffmpeg ffmpeg-devel -y || exit 1;
		# Assuming Fedora
		#sudo dnf install ffmpeg ffmpeg-devel -y || exit 1;

	fi

	for i in {1..53} ; do
		path=video/$i/${view}
		curr_files=`ls $path/ | cut -d"." -f1`
		#echo $curr_files
		#for file in $curr_files; do
		#	echo $path/$file.mp4
		#done
		#exit 1;
		for file in $curr_files; do
			#ffmpeg -i $path/$file.mp4 -vf fps=fps=30 -vf hue=s=0 ext_frames/$i/$file%03d.png || exit 1;   #stopped here
			ffmpeg -i $path/$file.mp4 -vf fps=fps=30  ext_frames${view}/$file%03d.png || exit 1;   #stopped here

		done

	done

fi


echo ===============
echo  " $0 ended at  `date` ";
exit 0;
