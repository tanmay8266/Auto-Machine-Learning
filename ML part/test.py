import sys
print("Hello PHP")
data = sys.argv[1]
list_of_data = data.split(";")
list_of_data.remove('')
print(list_of_data)
for i in range(len(list_of_data)):
    list_of_data[i] = list_of_data[i].split(":")[1]
    list_of_data[i] = [j for j in list_of_data[i].split(",")]
# [['9', '9', '9', '9', '9', '9', '9', '9', '9', '9', '9', '9', '9', '9'], ['7', '7', '7', '7', '7', '7', '7', '7', '7', '7', '7', '7', '7', '7']]
print(list_of_data)
data = []
for i in range(len(list_of_data[0])):
    data.append([item[i] for item in list_of_data])
print(data)
# data =  [['9', '7'], ['9', '7'], ['9', '7'], ['9', '7'], ['9', '7'], ['9', '7'], ['9', '7'], ['9', '7'], ['9', '7'], ['9', '7'], ['9', '7'], ['9', '7'], ['9', '7'], ['9', '7']]
#NAIVE BAYES
# age = ['Y','Y','M','S','S','S','M','Y','Y','S','Y','M','M','S']
# income = ['H','H','H','M','L','L','L','M','L','M','M','M','H','M']
# student = ['N','N','N','N','Y','Y','Y','N','Y','Y','Y','N','Y','N']
# credit = ['fair','excellent','fair','fair','fair','excellent','excellent','fair','fair','fair','excellent','excellent','fair','excellent']
# buysc = ['no','no','yes','yes','yes','no','yes','no','yes','yes','yes','yes','yes','no']
# data = [age,income,student,credit,buysc]
inputs = sys.argv[2].split(',')
print(inputs)
# inputs = ['9', '9', '9', '9', '9', '9', '9', '9', '9', '9', '9', '9', '9']
counts = []
ncounts = []
posOutcome, negOutcome = 0,0
for i in range(len(data)-1):
    # inputs.append(input())
    counts.append(0)
    ncounts.append(0)
print(inputs)
for i in range(len(data[0])):
    for j in range(len(data)-1):
        if(data[j][i] == inputs[j]):
            if(data[len(data)-1][i] == 'yes' or data[len(data)-1][i] == '1'):
                counts[j] += 1
            else:
                ncounts[j] += 1
    if(data[len(data)-1][i] == 'yes' or data[len(data)-1][i] == '1'):
        posOutcome += 1
    else:
        negOutcome += 1

y = posOutcome/len(data[0])
n = negOutcome/len(data[0])
print(posOutcome,negOutcome)
pred_y, pred_n = 1,1
for i in range(len(counts)):
    pred_y = pred_y * counts[i]
    pred_n = pred_n * ncounts[i]
print(pred_y,pred_n)
pred_y = pred_y/y**(len(inputs))
pred_n = pred_n/n**(len(inputs))
print(pred_y,pred_n)
if(pred_y >= pred_n):
    print("POSITIVE")
else:
    print("NEGATIVE")