# MIT Coupling study at Jupiter

This repository contains the implementation of the method used for the study of the Magnetosphere-Ionosphere-Thermosphere coupling at Jupiter using data from NASA's Juno mission. This code is associated to the publication *Magnetosphere-Ionosphere-Thermosphere Coupling Study at Jupiter Based on Juno’s First 30 Orbits and Modeling Tools*, by Al Saati et al. (2022, Journal of Geophysical Research - Space Physics, https://doi.org/10.1029/2022JA030586).

This work have been done during the master internship of Sariah Al Saati and Noé Clément at Institut de Recherche en Astrophysique et Planétologie (IRAP) in Toulouse, under the supervision of Michel Blanc and Nicolas André, in 2021.

The code has been partly adapted from the code of Yuxian Wang written in IDL written during his PhD at IRAP. 

## Contact
Sariah Al Saati - sariah.al-saati@polytechnique.edu

**Cite the paper:** [![DOI](https://zenodo.org/badge/DOI/10.1029/2022JA030586.svg)](https://doi.org/10.1029/2022JA030586)

**Cite the code:** [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.7023402.svg)](https://doi.org/10.5281/zenodo.7023402)

**Cite the data:** [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.7023034.svg)](https://doi.org/10.5281/zenodo.7023034)


# Data
All the data needed to run the code have been compressed into zip format and stored in the following zenodo repository under the DOI 10.5281/zenodo.7023034.

Please download the data and unzip it into the main folder before running the code. 

The current github repository must have the following structure (up to subfolder not indicated here):

```
├── Data
│   ├── ephemeris
│   ├── instruments
│   │   ├── jade
│   │   ├── jedi
│   │   ├── mag
│   │   ├── uvs
│   │   └── waves
│   ├── JunoSpiceKer
│   │   ├── fk
│   │   ├── lsk
│   │   ├── pck
│   │   ├── spk
│   │   └── waves
│   └── juno_posi.tm
├── Results
│   ├── MAGresiduals
│   ├── MODatmo
│   ├── MODelectro
│   ├── MODionos
│   └── PLT
├── *.py
├── LICENSE
├── requirements.txt
├── README.md
└── .gitignore
```


# Installation
When the repo is cloned, run 
```
$ pip install --upgrade -r requirements.txt
```
in your terminal to install all the necessary packages.

Examples of scripts used to run the code and to generate the plots showing the results of the study are given with the prefix 'PLT'.
