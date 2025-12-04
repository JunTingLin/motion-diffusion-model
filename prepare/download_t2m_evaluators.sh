echo -e "Downloading T2M evaluators"

# Download t2m.zip (with confirm=pbef to bypass virus scan warning)
echo "Downloading t2m.zip..."
gdown "https://drive.google.com/uc?export=download&confirm=pbef&id=1O_GUHgjDbl2tgbyfSwZOUYXDACnk25Kb" -O t2m.zip

# Download kit.zip
echo "Downloading kit.zip..."
gdown "https://drive.google.com/uc?export=download&confirm=pbef&id=12liZW5iyvoybXD8eOw4VanTgsMtynCuU" -O kit.zip

rm -rf t2m
rm -rf kit

unzip t2m.zip
unzip kit.zip
echo -e "Cleaning\n"
rm t2m.zip
rm kit.zip

echo -e "Downloading done!"
