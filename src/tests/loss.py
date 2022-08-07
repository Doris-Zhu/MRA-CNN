import matplotlib.pyplot as plt

indices = []
losses = []
with open('log/apn_1658742648.log', 'r') as f:
    content = f.readlines()[1:]
    for line in content:
        indices.append(len(indices) + 1)
        losses.append(float(line.split()[-1].strip()))

plt.plot(indices, losses)
plt.show()
