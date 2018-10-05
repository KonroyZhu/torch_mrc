import pickle

dev=pickle.load(open("dev1.pkl","rb"))
hit=0.0
for k in dev.keys():
    if dev[k]==0:
        hit+=1
print(hit/len(list(dev.keys())))