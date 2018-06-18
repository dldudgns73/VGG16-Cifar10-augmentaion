import matplotlib.pyplot as plt
import pickle as pkl

# base : 84.74 %
# augmentation1 : 87.11 %
# augmentation2 : 87.75 %

with open("train_model/base_model_scores.pkl", "rb") as f :
    scores = pkl.load(f)

plt.plot(scores)
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.title("VGG Augment model1")

fig = plt.gcf()
plt.show()
fig.savefig("augmentation1.jpg")