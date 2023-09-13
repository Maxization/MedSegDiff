import matplotlib.pyplot as plt
import numpy as np
from os import listdir
from os.path import isfile, join

def get_number(str):
  index = str.find(':') + 2
  str = str[index:].strip()

  return float(str)

path = 'E:/metrices'
cat = 'Greyscale'

onlyfiles = [join(path, f) for f in listdir(path) if isfile(join(path, f)) and cat.lower() in f]
print(onlyfiles)

iou_arr = []
precision_arr = []
recall_arr = []
accuracy_arr = []
dice_coef_arr = []
mse_arr = []

for file in onlyfiles:
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
      iou = get_number(lines[i+1])
      precision = get_number(lines[i+2])
      recall = get_number(lines[i+3])
      accuracy = get_number(lines[i+4])
      dice_coef = get_number(lines[i+5])
      mse = get_number(lines[i+6])
      i = i + 7

      split_iou.append(iou)
      split_precision.append(precision)
      split_recall.append(recall)
      split_accuracy.append(accuracy)
      split_dice_coef.append(dice_coef)
      split_mse.append(mse)

    iou_arr.append(split_iou)
    precision_arr.append(split_precision)
    recall_arr.append(split_recall)
    accuracy_arr.append(split_accuracy)
    dice_coef_arr.append(split_dice_coef)
    mse_arr.append(split_mse)

def draw_plot(array, name, category = ''):
  SMALL_SIZE = 16
  MEDIUM_SIZE = 18
  BIGGER_SIZE = 20

  plt.rc('font', size=SMALL_SIZE)  # controls default text sizes
  plt.rc('axes', titlesize=SMALL_SIZE)  # fontsize of the axes title
  plt.rc('axes', labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
  plt.rc('xtick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
  plt.rc('ytick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
  plt.rc('legend', fontsize=SMALL_SIZE)  # legend fontsize
  plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

  fig, ax = plt.subplots()
  ax.boxplot(array, labels=['Split1', 'Split2', 'Split3', 'Split4', 'Split5'])

  title = category + " " + name
  plt.title(title)
  plt.savefig("./plots/" + title.replace(' ', '_') + ".png")
  #plt.show()

draw_plot(iou_arr, 'IOU', cat)
draw_plot(precision_arr, 'Precision', cat)
draw_plot(recall_arr, 'Recall', cat)
draw_plot(accuracy_arr, 'Accuracy', cat)
draw_plot(dice_coef_arr, 'Dice Coef', cat)
draw_plot(mse_arr, 'MSE', cat)

# mean_arr = []
# for arr in dice_coef_arr:
#   mean_arr.append(np.median(arr))
#
# print(np.max(mean_arr))
