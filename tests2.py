# -*- coding: utf-8 -*-
"""
Created on Sat Dec 22 17:41:08 2018

@author: Sam
"""

import csv
facelist1 = []
facelist2 = []
true_negative = 0
false_positive = 0
true_positive = 0
false_negative = 0

with open('attribute_list.csv', 'rt', encoding = "ascii") as file1:
    with open('task_A.csv', 'r', encoding = "ascii") as file2:
        reader1 = csv.reader(file1)
        reader2 = list(csv.reader(file2))
        for row in reader1:
            if row[1] == "-1" and row[2] == "-1" and row[3] == "-1" and row[4] == "-1" and row[5] == "-1":
                facelist1.append(0)
            elif row[1] == "1" or row[2] == "1" or row[3] == "1" or row[4] == "1" or row[5] == "1":
                facelist1.append(1)
                
        for row in reader2:
            if ''.join(row).strip():
                if row[1] == "-1":
                    facelist2.append(0)
                else:
                    facelist2.append(1)

for i in range(len(facelist1)):
    if facelist1[i] + facelist2[i] == 2:
        true_positive += 1
    elif facelist1[i] > facelist2[i]:
        false_negative += 1
    elif facelist1[i] < facelist2[i]:
        false_positive += 1
    else:
        true_negative += 1
     
PPV = true_positive/(true_positive + false_positive) * 100
NPV = true_negative/(true_negative + false_negative) * 100
sensitivity = true_positive/(true_positive + false_negative) * 100
specificity = true_negative/(true_negative + false_positive) * 100        
print("\nTrue Positives:", true_positive,"\nFalse Positives:",false_positive,
      "\n\nTrue Negatives:",true_negative,"\nFalse Negatives:",false_negative)

print("\nPositive Predictive Value:",PPV,"\nNegative Predictive Value:",NPV,
      "\nSensitivity:",sensitivity,"\nSpecificity:",specificity)

