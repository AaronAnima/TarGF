# Install the *Room Rearrangement* Environment


To further run our *Room Rearrangement* environment, you need to further install `igibson`, download the preprocessed version of `3DFRONT` and meta-data used in our experiments following the instructions below:

First download the `3DFRONT` dataset preprocessed by us:
**[2022/8/31 update] Due to the license issue, we temporarily canceled the sharing link below. If you need this dataset urgently, please email the authors.**

```
cd Targf/envs/Room

wget https://www.dropbox.com/s/7j8f3dvn976hmaf/data.zip

unzip data.zip

rm -rf data.zip
```

Then setup softlink to `igibson` package and modify some modules of `igibson`:

```
pip install igibson==1.0.3

python setup_room_env.py # modify some files in igibson, and construct a softlink to data folder

```
Besides, you need to download the meta-data of our cleaned data:

```

wget https://www.dropbox.com/s/x6b2vuv8di8fyj8/RoomMetas.zip # download metadata

unzip RoomMetas.zip

rm -rf RoomMetas.zip

mkdir ../../ExpertDatasets # the metadata should be placed in the main path (TarGF)

cp RoomMetas/* ../../ExpertDatasets/ -r 

rm -rf RoomMetas RoomMetas.zip

cd ../../
```









