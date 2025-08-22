import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

import pandas as pd
from colorama import Fore
from Test import Test, make_checkpoint_name

def Test_resume(n, method, function, save_every):
    # Cartella checkpoint
    checkpoint_folder = 'Modified_checkpoints' if method == 'modified' else 'Truncated_checkpoints'
    name = make_checkpoint_name(n=n, method=method, function=function)
    checkpoint_csv = os.path.join(os.getcwd(), checkpoint_folder, f"{name}.csv")

    resume_from_comb = 0
    if os.path.exists(checkpoint_csv):
        try:
            df_tmp = pd.read_csv(checkpoint_csv)
            resume_from_comb = len(df_tmp)
            print(Fore.YELLOW + f"Riparto dalla combinazione {resume_from_comb} (da {checkpoint_csv})" + Fore.RESET)
        except Exception as e:
            print(Fore.RED + f"Impossibile leggere il checkpoint: {e}" + Fore.RESET)

    # Richiama Test con l’indice da cui riprendere
    Test(n, method, function, save_every, resume_from_comb=resume_from_comb)


if __name__ == '__main__':
    n = 10**5
    save_every = 15
    method = 'truncated'
    function = 'broyden_tridiagonal_function'
    Test_resume(n, method, function, save_every)