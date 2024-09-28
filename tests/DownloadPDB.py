import os
import requests

# Lista degli ID PDB di Set2

pdb_set_3b = [
    "1MEM", "1NLJ",
    "1PTY", "1G1F", "1JF7", "1NNY", "1ONZ", "1Q1M",
    "1BMQ",
    "1QS4"
]

# Directory dove salvare i file PDB
save_directory = 'c://temp/pdb_files/set3b'

# Creare la directory se non esiste
os.makedirs(save_directory, exist_ok=True)

def download_pdb(pdb_id):
    url = f'https://files.rcsb.org/download/{pdb_id}.pdb'
    response = requests.get(url)
    if response.status_code == 200:
        with open(os.path.join(save_directory, f'{pdb_id}.pdb'), 'wb') as file:
            file.write(response.content)
        print(f'Downloaded {pdb_id}.pdb')
    else:
        print(f'Failed to download {pdb_id}.pdb')

# Scaricare ogni file PDB di Set2
for pdb_id in pdb_set_3b:
    download_pdb(pdb_id)


pdb_set_1ids = [
    "1bid", "3tms", "1cdo", "8adh", "1dwd", "1hxf", "1fbp", "2fbp",
    "1gca", "1gcg", "1hew", "1hel", "1hyt", "1npc", "1inc", "1esa",
    "1rbp", "1brq", "1rob", "8rat", "1stp", "1swb", "1ulb", "1ula",
    "2ifb", "1ifb", "3ptb", "3ptn", "2ypi", "1ypi", "4dfr", "5dfr",
    "4phv", "3phv", "5cna", "2ctv", "7cpa", "8adh", "1a6w", "1a6u",
    "1apu", "3app", "1acj", "1qif", "1blh", "1djb", "1byb", "1bya",
    "1hfc", "1cge", "1ida", "1hsi", "1ivd", "1nna", "1mrg", "1ahc",
    "1mtw", "2tga", "1okm", "4ca2", "1pdz", "1pdy", "1phd", "1phc",
    "1pso", "1psn", "1qpe", "3lck", "1rne", "1bbs", "1snc", "1stn",
    "1srf", "1pts", "1stp", "2rta", "2ctc", "2ctb", "2h4n", "2cba",
    "2pk4", "1krn", "2sim", "2sil", "2tmn", "1l3f", "3gch", "1chg",
    "3mth", "6ins", "5p2p", "3p2p", "1imb", "1ime", "6rsa", "7rat"
]



pdb_ids_set2 = [
    "1g9v", "1gkc", "1gm8", "1gpk", "1hnn", "1hp0", "1hq2", "1hvy",
    "1hwi", "1hww", "1ia1", "1ig3", "1j3j", "1jd0", "1jje", "1jla",
    "1k3u", "1ke5", "1kzk", "1l2s", "1l7f", "1lpz", "1lrh", "1m2z",
    "1meh", "1mmv", "1mzc", "1n1m", "1n2j", "1n2v", "1n46", "1nav",
    "1of1", "1of6", "1opk", "1oq5", "1owe", "1oyt", "1p2y", "1p62",
    "1pmn", "1q1g", "1q41", "1q4g", "1r1h", "1r55", "1r58", "1r9o",
    "1s19", "1s3v", "1sg0", "1sj0", "1sq5", "1sqn", "1t40", "1t46",
    "1t9b", "1tow", "1tt1", "1tz8", "1u1c", "1u4d", "1uml", "1unl",
    "1uou", "1v0p", "1v48", "1v4s", "1vcj", "1w1p", "1w2g", "1x8x",
    "1xm6", "1xoq", "1xoz", "1y6b", "1ygc", "1yqy", "1yv3", "1yvf",
    "1ywr", "1z95", "2bm2", "2br1", "2bsm"
]

pdb_ids_3a = [
    "1O86",
    "1E66", "1GPK",
    "1PWL", "1PWM", "1T41",
    "1IEP",
    "1E1X", "1H07", "1H08", "1KE6", "1KE8", "1KE9",
    "4COX",
    "1KIJ", "1KZN",
    "1M17",
    "1C14", "1I2Z",
    "1EZQ",
    "1E9X", "1EA1",
    "1EET", "1EP4", "1FK9", "1RTH",
    "1TO3",
    "1DMP", "1HVR", "1HWR", "1QBS",
    "1HW8", "1HW9", "1HWI", "1HWJ", "1HWK", "1HWL",
    "1NF7",
    "1RV1",
    "1A4G", "1F8B", "1MWE", "1NNC", "2QWK",
    "1KV1",
    "1OYN", "1PTW", "1Q9M", "1RKO",
    "1UDT", "1UDU",
    "1QMF",
    "1KTS"
]

pdb_ids_3b = [
    "1MEM", "1NLJ",
    "1PTY", "1G1F", "1JF7", "1NNY", "1ONZ", "1Q1M",
    "1BMQ",
    "1QS4"
]