rm -r extracted ;
python consolidate.py ;
python assign.py ;
echo "Images assigned to subfigure labels." ;
python extract_images.py ;
rm -r assigned.json ;
echo "DONE"
