sh /netscratch/alghoul/install-pytorch.sh
echo "test ########## tr48 kitti 2015" >> ./fin_scripts/fin-tr48.txt
sh /netscratch/alghoul/code/StereoNet-Last-DFKI-V2/fin_scripts/fin-tr48-kitti2015.sh
echo "test ########## tr48 kitti 2012" >> ./fin_scripts/fin-tr48.txt
sh /netscratch/alghoul/code/StereoNet-Last-DFKI-V2/fin_scripts/fin-tr48-kitti2012.sh