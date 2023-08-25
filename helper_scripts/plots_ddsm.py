import matplotlib.pyplot as plt
import numpy as np
from os import listdir
from os.path import isfile, join

def get_number(str):
  index = str.find(':') + 2
  str = str[index:].strip()

  return float(str)

file = 'E:/results_ddsm.txt'

iou_arr = []
precision_arr = []
recall_arr = []
accuracy_arr = []
dice_coef_arr = []
mse_arr = []

with open(file) as f:
  lines = f.readlines()

  split_iou = []
  split_precision = []
  split_recall = []
  split_accuracy = []
  split_dice_coef = []
  split_mse = []

  i = 0
  while i < len(lines):
    print(lines[i])
    if lines[i].strip() == 'Average:':
      i = i + 7
      continue
    print(f"Iteration: {i}")
    name = lines[i].strip()
    iou = get_number(lines[i + 1])
    precision = get_number(lines[i + 2])
    recall = get_number(lines[i + 3])
    accuracy = get_number(lines[i + 4])
    dice_coef = get_number(lines[i + 5])
    mse = get_number(lines[i + 6])
    i = i + 7

    split_iou.append(iou)
    split_precision.append(precision)
    split_recall.append(recall)
    split_accuracy.append(accuracy)
    split_dice_coef.append(dice_coef)
    split_mse.append(mse)

def draw_plot(array, name):
  fig, ax = plt.subplots()
  ax.boxplot(array, labels=['Iou', 'Precision', 'Recall', 'Accuracy', 'Dice_coef', 'MSE'])

  title = name
  plt.title(title)
  plt.savefig("./plots/" + title.replace(' ', '_') + ".png")
  #plt.show()

array = [split_iou, split_precision, split_recall, split_accuracy, split_dice_coef, split_mse]
draw_plot(array, 'CBISDDSM density=1,2 metirces')


