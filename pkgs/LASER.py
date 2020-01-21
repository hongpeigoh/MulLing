import requests
import numpy as np
import subprocess
import platform #for OS name

address = '127.0.0.1:8050'

def check_if_laser_start():
    if isinstance(get_vect('Hello World'), (list, tuple, np.ndarray)):
        print('LASER loaded and ready to be used')
    else:
        raise ConnectionError('LASER port not set up. Please visit LASER\'s github to view the instructions in setting up the docker container.')

def get_vect(query_in, lang='en', address=address)
    url = 'http://' + address + '/vectorize'
    params = {'q':query_in, 'lang':lang}
    resp = requests.get(url=url, params=params).json()
    return resp['embedding']

if __name__ = "__main__":
    check_if_laser_start()