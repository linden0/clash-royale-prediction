
# ClashSense

## Description
This project has two goals:
- Classify verbal input of opponent's cards accurately
- Use the predictions to keep track of the opponent's cards and elixir
Essentially, as you are playing, speak the cards that your opponent plays (Giant, Knight, etc.) and the program will display your opponent's hand and elixir count. Here's a demo:


https://github.com/user-attachments/assets/b8ed995f-3ca9-4c1f-8057-b7e4809092e6


## How it Works
This project classifies audio using a PyTorch CNN on audio data represented by Mel-frequency cepstral coefficients (MFCCs). This classifer serves as the base for the realtime, continuous speech detection system. It takes audio segments, applies Voice Activity Detection, and if speech is detected, classifies the segment. 

The UI is built with Tkinter. For elixir and card tracking we use the following Clash Royale properties:
- If opponent plays card A, then plays 4 other cards, card A will be in their hand
- Elixir increments every 2.8 seconds

## Accuracy
The project achieves 100% validation accuracy, but is trained on my voice. You may see better results if you create your own dataset with `utils/collect.py` and retrain with `src/model/cnn.ipynb`.

## Usage
After cloning the project, create a virtual environment and install dependencies
```
python -m venv /path/to/new/virtual/environment
pip install -r requirements.txt
```
Run the project
```
python3 src/main.py
```


## Todo
* Add more classes for more cards
* Elixir count assumes single elixir only, add different rates for double and triple

    
