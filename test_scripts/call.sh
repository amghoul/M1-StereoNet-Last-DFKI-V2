sh /netscratch/alghoul/install.sh
echo "test ########## tr23" >> ./test_scripts/test-status.txt
sh /netscratch/alghoul/code/StereoNet-Last-DFKI-V2/test_scripts/test-tr23.sh >> ./test_scripts/test-status.txt
echo "test ########## tr25" >> ./test_scripts/test-status.txt
sh /netscratch/alghoul/code/StereoNet-Last-DFKI-V2/test_scripts/test-tr25.sh >> ./test_scripts/test-status.txt
echo "test ########## tr48" >> ./test_scripts/test-status.txt
sh /netscratch/alghoul/code/StereoNet-Last-DFKI-V2/test_scripts/test-tr48.sh >> ./test_scripts/test-status.txt
echo "test ########## tr52" >> ./test_scripts/test-status.txt
sh /netscratch/alghoul/code/StereoNet-Last-DFKI-V2/test_scripts/test-tr52.sh >> ./test_scripts/test-status.txt
echo "test ########## finished all" >> ./test_scripts/test-status.txt