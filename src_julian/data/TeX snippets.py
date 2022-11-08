# how to plot an image with matplotlib and save to Seafile folder
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use("pgf")
matplotlib.rcParams.update({
    "pgf.texsystem": "pdflatex",
    'font.family': 'serif',
    'text.usetex': True,
    'pgf.rcfonts': False,
})
plt.figure()
plt.scatter(10, 10)
plt.title('This is my title')

path_to_exchange_folder = '/mnt/ssd/julian/documentation/Meine Bibliothek/My_Exchange/'
filename = 'testfile'
plt.savefig(path_to_exchange_folder + filename + '.pgf')