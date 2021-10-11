sh /netscratch/alghoul/install-pytorch.sh
echo "test ########## tr23 kitti 2015" >> ./fin_scripts/fin-tr23.txt
sh /netscratch/alghoul/code/StereoNet-Last-DFKI-V2/fin_scripts/fin-tr23-kitti2015.sh
echo "test ########## tr23 kitti 2012" >> ./fin_scripts/fin-tr23.txt
sh /netscratch/alghoul/code/StereoNet-Last-DFKI-V2/fin_scripts/fin-tr23-kitti2012.sh