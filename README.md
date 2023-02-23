# Compression Project

This is the exercise code for Compression Data using AutoEncoders.

## How to run the code
Per runnare il codice:
1. Installa tutte le dipendenze u **pip install -r requirements.txt**.
2. Run the code: **python3 main.py**, puoi cambiare i parametri o direttamente dal file yaml contenuto nella cartella **cfg/config.yaml** oppure direttamente da linea di comando, esempi **python3 main.py save_path= "ciccio"**. In questo modo ho cambiato il nome della cartella di salvataggio.
3. Prima bisogna eseguire una run di train: **python3 main.py train=True** e successivamente una run di test **python3 main.py train=False**