# Photoacoustic Image Quality Assessment  

## About  
This repository contains code and data for my MSc AI thesis project. The aim is to evaluate the quality of photoacoustic images using a combination of traditional metrics (PSNR, SSIM) and no-reference metrics (BRISQUE, NIQE). Deep learning-based methods will also be explored to enhance image quality assessment.  

## Project Goals  
1. Analyze various optoacoustic array configurations (semi-circle, multisegment, linear, and virtual circle).  
2. Evaluate image quality across configurations using four key metrics.  
3. Develop and test deep learning-based image quality models.  

## Datasets  
- OADAT dataset (experimental and synthetic optoacoustic data).  
- Mouse imaging dataset (optical and acoustic data for reconstruction).  

## Repository Structure  
- `src/`: Code for metrics computation and deep learning models.  
- `data/`: (Ignored) Raw and processed datasets.  
- `notebooks/`: Jupyter notebooks for analysis and visualization.  
- `results/`: Generated tables, graphs, and model outputs.  
- `docs/`: Thesis write-up and related documentation.  

## Setup and Usage  
1. Clone this repository.  
2. Set up the environment using `requirements.txt`.  
3. Run scripts in `src/` to compute metrics and analyze datasets.  

## License  
[MIT License](LICENSE)

## Contact  
For any questions or collaboration inquiries, contact me at melle.vanderbrugge@student.uva.nl

## Run examples
python src/main.py --datasets SCD
python src/main.py --datasets SWFD --file_key multisegment
python src/main.py --datasets SWFD --file_key semicircle
python src/main.py --datasets MSFD
python src/main.py --datasets mice
python src/main.py --datasets phantom
python src/main.py --datasets v_phantom