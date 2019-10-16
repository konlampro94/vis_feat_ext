#! /bin/bash


if [ ! -d "video"  ] ; then
	read -p "Please give me the relative path to the cropped_mouth_mp4_phrase video folder: " rel_path
	: "${rel_path:=../../OuluVS2/cropped_mouth_mp4_phrase}"
	echo "Linking cropped_mouth_mp4_phrase here"
	ln -s $rel_path video ||  exit 1;
	
fi

echo "Argument checking..."

if [[ "$#" -ge 1 ]] ; then 
	view=$1
	mode=$2
else
	read -p "Please give me view(1-5) to upsample: " view
	read -p "Please give me the which data do you want (train/test): " mode
fi

#folder=frames_100fps
 

echo View specified : "$view"
echo Mode specified : "$mode"

echo "Sleeping for 2s!"
sleep 2
folder="$mode$view" || exit 1;
echo  $folder
sleep 2

if [ ! -d "$folder" ] ; then

	mkdir "$folder" || exit 1;

	files=$(ls video)
	#echo $files
	#file_front_mp4=$(for file in $files ; do  ls video/$file/1 ;  done )
	#echo $file_front_mp4

	if [ "$mode" = train ] ; then
		for i in {1..41} ; do
			#mkdir $folder/5
			path=video/$i/$view
			curr_files=`ls $path/ | cut -d"." -f1`
			#echo $curr_files
			#for file in $curr_files; do
			#	echo $path/$file.mp4
			#done
			#exit 1;
			for file in $curr_files; do
				#ffmpeg -i $path/$file.mp4 -vf fps=fps=100 -vf hue=s=0 $folder/$i/$file%03d.png || exit 1;   #stopped here
				mkdir $folder/$file
				ffmpeg -i $path/$file.mp4 -vf fps=fps=96  $folder/$file/$file%03d.png || exit 1;   #stopped here

			done

		done
	
	elif [ "$mode" = test ]
	then
		for i in {42..53} ; do
			#mkdir $folder/5
			path=video/$i/$view
			curr_files=`ls $path/ | cut -d"." -f1`
			#echo $curr_files
			#for file in $curr_files; do
			#	echo $path/$file.mp4
			#done
			#exit 1;
			for file in $curr_files; do
				#ffmpeg -i $path/$file.mp4 -vf fps=fps=100 -vf hue=s=0 $folder/$i/$file%03d.png || exit 1;   #stopped here
				mkdir $folder/$file
				ffmpeg -i $path/$file.mp4 -vf fps=fps=96  $folder/$file/$file%03d.png || exit 1;   #stopped here

			done

		done 

	else
		echo "Mode error!!!!!!"
		exit 1;
	fi
fi


echo ===============
echo  " $0 ended at  `date` ";
exit 0;
