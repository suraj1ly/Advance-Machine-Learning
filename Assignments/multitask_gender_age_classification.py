
# coding: utf-8

# In[1]:

# Reference : https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
# Reference : https://github.com/ivangrov/Datasets-ASAP/blob/master/%5BPart%201%5D%20IMDB-WIKI/Read_IMDB_WIKI_Dataset.ipynb

import scipy.io
import numpy as np
import os
import pandas as pd
import pickle
import seaborn as sns
import torch.optim as optim
import matplotlib.pyplot as plt
import torch
import  datetime
from shutil import copy2
import cv2
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
from scipy.io import loadmat
import torch.nn.functional as F
from sklearn.decomposition import PCA
from torch.utils.data.sampler import SubsetRandomSampler
import warnings
warnings.filterwarnings('ignore')


# In[2]:

# Initialisation of Parameters
epoch_train = 15
global dropout 
dropout = 0.2
learning_rate = 0.001 
momentum = 0.9
init_weights = "xavier"
global padding 
global activation_function
activation_function = "relu"
padding = 1
batch_size = 32
quantile_term = 10


# # Reading the Data for Wiki Face 

# In[3]:


location= "./data/wiki_crop/wiki.mat"
wikiMat = scipy.io.loadmat(location)
wikiPlace = wikiMat['wiki'][0][0]


# In[4]:

def age_class_predict(age,quantile_term):
    return age/quantile_term


# In[5]:

place = wikiPlace
relative_add = "./data/wiki_crop/"
total = 0
dataset = []
gender_class = []
age_class = []
for i in range(62328):
    if i % 10000 ==0:
        print(i)
  
    bYear = int(place[0][0][i]/365) #birth year
    taken = place[1][0][i] #photo taken
    path = place[2][0][i][0] #image_path
    gender = place[3][0][i] # Female/Male
    name = place[4][0][i] # Name
    faceBox= place[5][0][i] # Face coords
    faceScore = place[6][0][i] #Face score
    secFaceScore = place[7][0][i] #Sec face score

    #Calculating age
    age = taken - bYear
    faceScore = str(faceScore)
    secFaceScore = str(secFaceScore)
    if 'n' not in faceScore: # n as in Inf; if true, implies that there isn't a face in the image

        if 'a' in secFaceScore: #a as in NaN; implies that no second face was found

            if age >= 0 and age <= 100: 

                try:
                    gender = int(gender)
                    total += 1
                    temp=[]
                    img = cv2.imread(relative_add + str(path))
                    img = cv2.resize(img,(32,32))
                    dataset.append(img.reshape(3,32,32))
                    gender_class.append(gender)
                    age_class.append(age_class_predict(age,quantile_term))
                except:
                    continue


# In[28]:

np.ndarray.dump(np.array(dataset),"wiki_face_data.npy")
np.ndarray.dump(np.array(gender_class),"wiki_face_gender_label.npy")
np.ndarray.dump(np.array(age_class),"wiki_face_age_label.npy")
# dataset = np.load("wiki_face_data.npy")
# gender_class = np.load("wiki_face_gender_label.npy")
# age_class = np.load("wiki_face_age_label.npy")
# dataset = np.array(dataset)


# In[ ]:




# In[83]:

np.save("wiki_face_data.npy",np.array(dataset))
np.save("wiki_face_gender_label.npy",np.array(gender_class))
np.save("wiki_face_age_label.npy",np.array(age_class))


# In[ ]:




# # Visualisation of Dataset

# In[30]:

def pca_plot(dataset_main,label_main,title,classes):
    #TSNE Plot for glass dataset
    tsne = PCA(n_components=2)
    tsne_results = tsne.fit_transform(dataset_main)

    df_subset = pd.DataFrame()
    df_subset['X'] = tsne_results[:,0]
    df_subset['y']=label_main
    df_subset['Y'] = tsne_results[:,1]
    plt.figure(figsize=(6,4))
    plt.title(title)
    sns.scatterplot(
        x="X", y="Y",
        hue="y",
        palette=sns.color_palette("hls", classes),
        data=df_subset,
        legend="full",
        alpha=1.0
    )


# In[ ]:

train_data_main = []
for i in dataset:
    train_data_main.append(np.array(i).ravel())


# In[ ]:

pca_plot(train_data_main,gender_class,"Visualisation for Wiki Face  Datset",2)


# In[ ]:

#Dataset description for Glass dataset
dict_label={}
for i in range(len(gender_class)):
    try:
        dict_label[gender_class[i]] = dict_label[gender_class[i]]+1
    except:
        dict_label[gender_class[i]]=1
print("Class distribution : ",dict_label)
labels = list(dict_label.keys())
index = np.arange(len(list(dict_label.keys())))
plt.pie(list(dict_label.values()),labels=labels,autopct='%1.0f%%')
labels = list(dict_label.keys())
plt.title("Class Distribution in Wiki Face dataset")
plt.show()


# In[ ]:




# In[ ]:




# In[14]:

# Reference : https://stackoverflow.com/questions/50052295/how-do-you-load-images-into-pytorch-dataloader
def load_dataset():
    data_path = 'data/train/'
    train_dataset = torchvision.datasets.ImageFolder(
        root=data_path,
        transform=torchvision.transforms.ToTensor()
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=64,
        num_workers=0,
        shuffle=True
    )
    return train_loader


# In[ ]:




# In[31]:




# In[ ]:




# In[6]:

train_size = 30000
tensor_X = torch.stack([torch.from_numpy(i) for i in dataset])
tensor_gender = torch.stack([torch.from_numpy(np.array(i)) for i in gender_class])
tensor_age = torch.stack([torch.from_numpy(np.array([i])) for i in age_class])
X_train = tensor_X[:train_size]
gender_train = tensor_gender[:train_size]
age_train = tensor_age[:train_size]
X_test = tensor_X[train_size:]
gender_test = tensor_gender[train_size:]
age_test = tensor_age[train_size:]
train_data = torch.utils.data.TensorDataset(X_train, gender_train, age_train)
train_indices = np.arange(len(train_data))
np.random.shuffle(train_indices.tolist())
train_sample = SubsetRandomSampler(train_indices)
trainloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, sampler=train_sample, num_workers=0)
test_dataset = torch.utils.data.TensorDataset(X_test, gender_test, age_test)
testloader = torch.utils.data.DataLoader(test_dataset)


# In[ ]:




# In[11]:

# Reference : https://stackoverflow.com/questions/49433936/how-to-initialize-weights-in-pytorch
def init_weights_xavier(m):
    if type(m) == nn.Linear or type(m) == nn.Conv2d:
        torch.nn.init.xavier_uniform(m.weight)
        m.bias.data.fill_(0.01)
        
def init_weights_normal(m):
        classname = m.__class__.__name__
        # for every Linear layer in a model
        if classname.find('Linear') == -1:
            pass
        else:
            y = m.in_features
            m.weight.data.normal_(0.0,np.sqrt(1/y))
            m.bias.data.fill_(0)
        if classname.find('Conv2d') == -1:
            pass
        else:
            y = m.kernel_size[0]*m.kernel_size[1]*m.out_channels
            m.weight.data.normal_(0.0,np.sqrt(2/y))
            m.bias.data.fill_(0)


# In[12]:

len(list(set(age_class)))


# In[13]:

class Net_Q4(nn.Module):
    def __init__(self):
        global dropout
        global padding
        global activation_function
        super(Net_Q4, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3,padding=padding)
        self.batchnorm1 = nn.BatchNorm2d(16)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16,32, 3,padding=padding)
        self.batchnorm2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32,64, 3,padding=padding)
        self.batchnorm3 = nn.BatchNorm2d(64)
        ##### 
        self.conv4 = nn.Conv2d(64,128, 3,padding=padding)
        self.batchnorm4 = nn.BatchNorm2d(128)
        self.conv5 = nn.Conv2d(128,256, 3,padding=padding)
        self.batchnorm5 = nn.BatchNorm2d(256)
        self.conv6 = nn.Conv2d(256,512, 3,padding=padding)
        self.batchnorm6 = nn.BatchNorm2d(512)
        #####
        self.conv7 = nn.Conv2d(64,128, 3,padding=padding)
        self.batchnorm7 = nn.BatchNorm2d(128)
        self.conv8 = nn.Conv2d(128,256, 3,padding=padding)
        self.batchnorm8 = nn.BatchNorm2d(256)
        self.conv9 = nn.Conv2d(256,512, 3,padding=padding)
        self.batchnorm9 = nn.BatchNorm2d(512)
        ######
        self.dropout = nn.Dropout(dropout)
        self.fc1_age = nn.Linear(512 * 4 * 4, 100)
        self.fc2_age = nn.Linear(100, 1)
        self.fc1_gender = nn.Linear(512 * 4 * 4, 100)
        self.fc2_gender = nn.Linear(100, 2)

    def forward(self, x):
        if activation_function == "relu":
            x = self.dropout(self.pool(F.relu(self.batchnorm1(self.conv1(x)))))
            x = self.dropout(F.relu(self.batchnorm2(self.conv2(x))))
            x = self.dropout(self.pool(F.relu(self.batchnorm3(self.conv3(x)))))
            
            #####
            x1 = self.dropout(F.relu(self.batchnorm4(self.conv4(x))))
            x1 = self.dropout(self.pool(F.relu(self.batchnorm5(self.conv5(x1)))))
            x1 = self.dropout(F.relu(self.batchnorm6(self.conv6(x1))))
            x1 = x1.view(-1, 512 * 4 * 4)
            x_gender = self.dropout(F.relu(self.fc1_gender(x1)))
            x_gender = F.softmax(self.fc2_gender(x_gender))
            
            #####
            x2 = self.dropout(F.relu(self.batchnorm7(self.conv7(x))))
            x2 = self.dropout(self.pool(F.relu(self.batchnorm8(self.conv8(x2)))))
            x2 = self.dropout(F.relu(self.batchnorm9(self.conv9(x2))))
            x2 = x2.view(-1, 512 * 4 * 4)
            x_age = self.dropout(F.relu(self.fc1_age(x2)))
            x_age = F.relu(self.fc2_age(x_age))
            
        elif activation_function == "tanh":
            x = self.dropout(self.pool(F.relu(self.batchnorm1(self.conv1(x)))))
            x = self.dropout(F.relu(self.batchnorm2(self.conv2(x))))
            x = self.dropout(self.pool(F.relu(self.batchnorm3(self.conv3(x)))))
            
            #####
            x1 = self.dropout(F.relu(self.batchnorm4(self.conv4(x))))
            x1 = self.dropout(self.pool(F.relu(self.batchnorm5(self.conv5(x1)))))
            x1 = self.dropout(F.relu(self.batchnorm6(self.conv6(x1))))
            x1 = x1.view(-1, 512 * 4 * 4)
            x_gender = self.dropout(F.relu(self.fc1_gender(x1)))
            x_gender = F.softmax(self.fc2_gender(x_gender))
            
            #####
            x2 = self.dropout(F.relu(self.batchnorm7(self.conv7(x))))
            x2 = self.dropout(self.pool(F.relu(self.batchnorm8(self.conv8(x2)))))
            x2 = self.dropout(F.relu(self.batchnorm9(self.conv9(x2))))
            x_age = self.dropout(F.relu(self.fc1_age(x2)))
            x_age = F.relu(self.fc2_age(x_age))
        return x_gender, x_age




# In[ ]:




# In[14]:

net = Net_Q4()
if init_weights == "xavier":
    net.apply(init_weights_xavier)
if init_weights == "random":
    net.apply(init_weights_normal)
net = net.cuda()


# In[15]:

# Define a Loss function and optimizer
criterion_gender = nn.CrossEntropyLoss()
criterion_age = nn.MSELoss()
optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=momentum)
size_train = len(trainloader.dataset)
size_test = len(testloader.dataset)


# In[16]:

train_loss = []
accuracy = []
gradients = {}
gradients_temp = {}
for epoch in range(epoch_train):  # loop over the dataset multiple times
    
    running_loss = 0.0
    counter  = 0 
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, label_gender,label_age = data
        inputs = inputs.cuda()
        label_gender = label_gender.cuda()
        label_age = label_age.cuda()
        label_age = label_age.float()
        label_gender = label_gender.long()
        inputs = inputs.float()
        # zero the parameter gradients
        optimizer.zero_grad()
        # forward + backward + optimize
        
        output_gender,output_age = net(inputs)
#         output_gender.squeeze(1)
        loss_gender = criterion_gender(output_gender, label_gender)
        loss_age = criterion_age(output_age, label_age)
        loss = loss_gender + loss_age
        loss.backward()
        
        for n,p in net.named_parameters():
            if n.split('.')[1] =="weight" and n.split('.')[0][:-1] != "batchnorm":
                try:
                    gradients_temp[n.split('.')[0]].append(np.linalg.norm(p.grad))
                except:
                    temp = []
                    temp.append(np.linalg.norm(p.grad))
                    gradients_temp[n.split('.')[0]] = temp 
                    
        if  counter == (size_train/batch_size -1):
        
            for n,p in net.named_parameters():
                if n.split('.')[1] =="weight" and n.split('.')[0][:-1] != "batchnorm":
                    try:
                        gradients[n.split('.')[0]].append(np.sum(gradients_temp[n.split('.')[0]]))
                    except:
                        gradients[n.split('.')[0]] = [np.sum(gradients_temp[n.split('.')[0]])]
            gradients_temp = {}

             
        optimizer.step()
        counter = counter + 1
        # print statistics
        
        running_loss += loss.item()
#         print("running_loss ",running_loss)
    if epoch%10 == 0:
        print("Trained till epoch ",epoch)
        print("Training loss : ",running_loss)
    train_loss.append(running_loss)
print('Finished Training')


# In[ ]:

plt.title("Plot for Loss")
plt.plot([j for j in range(len(train_loss))],train_loss)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.show()


# In[ ]:


for i in gradients.keys():
    plt.title("Plot for epoch vs gradients for blocks")
    plt.plot([j for j in range(len(gradients[i]))],gradients[i],label = i)
    plt.xlabel("Epoch")
    plt.ylabel("Gradients")
    plt.legend()
plt.show()


# In[ ]:

def accuracy_prediction(loader,net):
    # For Training
    correct_age = 0
    correct_gender = 0
    total = 0
    with torch.no_grad():
        for data in loader:
            images, label_gender,label_age = data
            images = images.cuda()
            label_gender = label_gender.cuda()
            label_age = label_age.cuda()
            label_age = label_age.long()
            label_gender = label_gender.long()
            images = images.float()
            output_gender,output_age = net(images)
            _, predict_gender = torch.max(output_gender.data, 1)
            _, predict_age = torch.max(output_age.data, 1)
            total += label_gender.size(0)
#             print(predict_age)
            correct_age  = correct_age + (predict_age - label_age)
            correct_gender += (predict_gender == label_gender).sum().item()
            
    test_accuracy =[] 
    test_accuracy.append((correct_age / float(total)) *100)
    test_accuracy.append((correct_gender / float(total)) *100)
    
    return test_accuracy


# In[ ]:

accuracy_prediction(trainloader,net)


# In[ ]:

accuracy_prediction(testloader,net)


# In[ ]:

# transform = transforms.Compose(
#     [transforms.ToTensor(),
#      transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# trainset = torchvision.datasets.STL10(root='./data', train=True,
#                                         download=True, transform=None)
# trainloader = torch.utils.data.DataLoader(trainset, batch_size=64,
#                                           shuffle=True, num_workers=10)

# testset = torchvision.datasets.STL10(root='./data', train=False,
#                                        download=True, transform=transform)
# testloader = torch.utils.data.DataLoader(testset, batch_size=64,
#                                          shuffle=False, num_workers=10)

# classes = ('plane', 'car', 'bird', 'cat',
#            'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


# In[ ]:



