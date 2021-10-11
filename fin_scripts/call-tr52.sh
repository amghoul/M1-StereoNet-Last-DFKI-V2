sh /netscratch/alghoul/install-pytorch.sh
echo "test ########## tr52 kitti 2015" >> ./fin_scripts/fin-tr52.txt
sh /netscratch/alghoul/code/StereoNet-Last-DFKI-V2/fin_scripts/fin-tr52-kitti2015.sh
echo "test ########## tr52 kitti 2012" >> ./fin_scripts/fin-tr52.txt
sh /netscratch/alghoul/code/StereoNet-Last-DFKI-V2/fin_scripts/fin-tr52-kitti2012.sh