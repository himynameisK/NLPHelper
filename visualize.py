# Import libraries
from matplotlib import pyplot as plt
import numpy as np

# Creating dataset
def create_image(data_array):
    cars = ['positive', 'negative', 'neutral']

    # Creating plot
    fig = plt.figure(figsize=(10, 7))
    plt.pie(data_array, labels=cars)
    plt.title('Статистика восприятия видео:')
    plt.savefig('visual.png')