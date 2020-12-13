# Using clustering for pulmonary disease detection through respiratory sound analysis :loud_sound:
This repository contains the implementation of the related article that was written as a part of Machine learning course at the Faculty of Computer and Information Science at the University of Ljubljana.

Related article: [Using clustering for pulmonary disease detection through respiratory sound analysis](https://github.com/lzontar/Clustering-Respiratory-Sounds/blob/master/article/Using-clustering-for-pulmonary-disease-detection-through-respiratory-sound-analysis.pdf)

# Repo structure :blue_book:
This repository contains folders:
* ```images/``` - contains all the visualizations and results of our work.
* ```library/``` - contains Python helper files that are used in the main Python script ```main.py```.
* ```results/``` - contains the ```data.json``` file, where we exported data from Kaggle in a more discrete way. It contains calculated features that are later used for clustering.
* ```article/``` - contains the article: [Using clustering for pulmonary disease detection through respiratory sound analysis](https://github.com/lzontar/Clustering-Respiratory-Sounds/blob/master/article/Using-clustering-for-pulmonary-disease-detection-through-respiratory-sound-analysis.pdf).
* ```main.py``` - the main Python script used to generate results. 

# Reproducing results :snake:
To reproduce my results, you will have to download the dataset I used from [Kaggle](https://www.kaggle.com/vbookshelf/respiratory-sound-database). Unzip it to folder ```data/``` in the root of the repository.

After you successfully forked this repo and downloaded the dataset, you will have to install Python dependencies:
```
pip install -r requirements.txt
```
Since I also used a library that is accesible through ```pip``` command, you will also have to execute the following set of instructions:
```
git clone https://github.com/raphaelvallat/entropy.git library/entropy/
cd library/entropy/
pip install -r requirements.txt
python setup.py develop
```
Now your environment is ready to go. Executing ```main.py``` produces some interesting results. :partying_face: :clinking_glasses:
