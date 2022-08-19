import time
import os
import random
from Loss import ContrastiveLoss
from model import VGG16Model
from torchvision import transforms
import torch
import sys
from tqdm import tqdm
from DataSet import BatchDataset
import matplotlib.pyplot as plt

torch.backends.cudnn.benchmark=True
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if len(sys.argv)>1:
    #output file name for model parameter
    model_path=sys.argv[1]

def transform(sample):
   
    trans = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((image_size1,image_size2)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, ), (0.5, )),
    ])
   
    pos=trans(torch.Tensor(sample["pos"]))
    neg=trans(torch.Tensor(sample["neg"]))
    anchor=trans(torch.Tensor(sample["anchor"]))
    
    sample = {"anchor":anchor,"pos":pos,"neg":neg}
    return sample

def train(model,dataloaders,optimizer,criterion,scheduler,epochs=10):
    train_loss=[]
    valid_loss=[]
    scaler = torch.cuda.amp.GradScaler()
    print(time.strftime('%Y/%m/%d %H:%M:%S'))

    for epoch in range(epochs):
        for phase in["train","valid"]:
            if phase=="train":
                model.train()
            else:
                model.eval()
            epoch_loss = 0.0
            for batch_data in tqdm((dataloaders[phase])):                    
                optimizer.zero_grad()   
                with torch.set_grad_enabled(phase=="train"):
                    with torch.cuda.amp.autocast():
                        anchor, pos, neg=batch_data["anchor"].to(device),batch_data["pos"].to(device),batch_data["neg"].to(device)
                        batch_emb_anchor=model(anchor)
                        batch_emb_pos=model(pos)
                        batch_emb_neg=model(neg)
                        
                        loss = criterion(batch_emb_anchor,batch_emb_pos,batch_emb_neg)
                        
                    if phase=="train":
                        scaler.scale(loss).backward()
                        scaler.step(optimizer)
                        scaler.update()
                    epoch_loss+=loss.item()*batch_data["pos"].size(0)
            epoch_loss=epoch_loss/len(dataloaders[phase].dataset)  
            if phase=="train":
                train_loss.append(epoch_loss)
                scheduler.step()
                torch.save(model.state_dict(), model_path)
            else:
                valid_loss.append(epoch_loss) 

            if epoch>0:
                plt.plot(train_loss,label="train")
                plt.plot(valid_loss,label="valid")
                plt.xlabel("epochs")
                plt.ylabel("loss")
                plt.legend()
                plt.savefig("loss.png",dpi=200, format="png")
                plt.clf()
            print("Time {} | Epoch {}/{}|{:5}  | Loss:{:.4f} ".format(time.strftime('%Y/%m/%d %H:%M:%S'),epoch+1,epochs,phase,epoch_loss))
    print(time.strftime('END:%Y/%m/%d %H:%M:%S'))

if __name__=="__main__":
    path="../data"
    category_key=[i for i in os.listdir(path) ]
    category_id=[i for i in range(len(category_key))]
    category={}
    for i,j in zip(category_key,category_id):
        category[i]=j
    path=[os.path.join(path,i) for i in os.listdir(path) ]
    files=[os.path.join(i,j) for i in path for j in os.listdir(i)]
    data=[]
    for file in files:
        data.append([file,category[file.split("\\")[1]]])
    
    image_size1,image_size2=1080//8,1920//8   
    random.seed(100)
    random.shuffle(data)
    n_train=int(len(data)*0.8)
    n_valid=int(len(data)*0.1)
    n_test=int(len(data)-n_train-n_valid)

    d=[]
    for i in data:
        d.append(i[0])

    train_dataset = BatchDataset(
        files=d[:n_train],
        
        transform=transform
        )
    valid_dataset = BatchDataset(
        files=d[n_train:n_train+n_valid],
    
        transform=transform
        )

    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=64,
        shuffle=True,
        num_workers=4
        )
    valid_loader = torch.utils.data.DataLoader(
        dataset=valid_dataset,
        batch_size=32,
        shuffle=True,
        num_workers=4
        )

    model=VGG16Model()
    model.to(device)
    lr=1e-5
    epochs=10
    criterion= ContrastiveLoss().to(device)
    optimizer=torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30,80], gamma=0.1)
    dataloaders=dict({"train":train_loader,"valid":valid_loader})

    train(model,dataloaders,optimizer,criterion,scheduler,epochs)
    