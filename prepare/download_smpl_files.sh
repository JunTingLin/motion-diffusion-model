mkdir -p body_models
cd body_models/

echo -e "The smpl files will be stored in the 'body_models/smpl/' folder\n"
gdown "https://drive.google.com/uc?export=download&confirm=pbef&id=1INYlGA76ak_cKGzvpOV2Pe6RkYTlXTW2" -O smpl.zip
rm -rf smpl

unzip smpl.zip
echo -e "Cleaning\n"
rm smpl.zip

echo -e "Downloading done!"