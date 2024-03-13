import statistics

def variance(data):
  n = len(data)
  mean = sum(data) / n
  deviations = [(x - mean) ** 2 for x in data]
  variance = sum(deviations) / n
  return variance
 
def stdev(data):
  import math
  var = variance(data)
  std_dev = math.sqrt(var)
  return std_dev


tab = [1, 2, 2.4, 2.6]
m = statistics.mean(tab)
s = stdev(tab)

print(m-2*s)
print(m+2*s)

tab = [1, 2, 2.4, 2.6, 1]
m = statistics.mean(tab)
s = stdev(tab)

print(m-2*s)
print(m+2*s)

