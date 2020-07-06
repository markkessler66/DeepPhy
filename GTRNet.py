import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
import os



def unix2win(folder):
    
    for filename in os.listdir(folder):
       infilename = os.path.join(folder,filename)
       if not os.path.isfile(infilename): continue
       oldbase = os.path.splitext(filename)
       newname = infilename.replace('.dat', '.dat')
       output = os.rename(infilename, newname)

unix2win('data')

def determine_seq_len(datafilename):
    myfile = open('data/' + datafilename, 'r')
    lines = myfile.readlines()
    seq_chars = ['A','C','G','T']
    counter = 0
    for char in lines[1]:
        if char in seq_chars:
            counter += 1
    return counter

seq_len = determine_seq_len(os.listdir('data')[0])

DEVICE = torch.device('cuda:0')
TRAINING_DATA_FOLDER = 'E:/TrainingData/training_Data_gtr.npy' ### PUT DESIRED FILE LOCATION HERE <ALLOW FOR 10GB>

filelist = os.listdir('data')
uno = []
dos = []
tres = []

for file in filelist:
    if file[0]=='a':
        uno.append(file)
    elif file[0]=='b':
        dos.append(file)
    else:
        tres.append(file)



filenames = {'filename1':'alpha_seqs.dat', 'filename2':'beta_seqs.dat', 'filename3' :'charlie_seqs.dat'}

seq_len = 200

num_trees = 7000

def combine_files(filelist, treename):
    newfile = open(f'{treename}_seqs.dat', 'a')

    for myfile in filelist:
        file1 = open('data/' + myfile, 'r')
        newfile.writelines(file1.readlines())
        file1.close()
        
    
    newfile.close()

combine_files(uno, 'alpha')
combine_files(dos, 'beta')
combine_files(tres, 'charlie')

def rebuild(matrix, tree_name):
    rebuilt = np.zeros((4,4*seq_len))
    if tree_name == filenames['filename2']:
        exchange = {0:0,1:3,2:2,3:1}
    elif tree_name == filenames['filename3']:
        exchange = {0:0,1:2,2:1,3:3}
    else:
        return matrix
        
    for row in range(3):
        rebuilt[row] = matrix[exchange[row]]
    
    return rebuilt

def hot(line):
    """
    Given a line in a .txt or .dat file, hotencode A,C,G,Ts in the line
    """
    out = []
    encoder_dict ={'A':[1,0,0,0], 'T':[0,1,0,0], 'G':[0,0,1,0], 'C':[0,0,0,1]}
    for item in line:
        if item in encoder_dict.keys():
            out.append(encoder_dict[item])
    
    return out

def link(seq_of_seqs):
    master_list =[]
    for list1 in seq_of_seqs:
        for item in list1:
            master_list.append(item)
    
    return master_list

REBUILD_DATA = True


class SequenceDataset():
    TREE1 = filenames['filename1']
    TREE2 = filenames['filename2']
    TREE3 = filenames['filename3']

    TREELIST = [TREE1,TREE2,TREE3]
    LABELS = {TREE1:0, TREE2:1, TREE3:2}
    training_data = []

    def make_training_data(self):
        for label in self.LABELS:
            myfile = open(label,'r')
            Lines = myfile.readlines()
            counter = 0
            set_of_seqs =np.zeros((4, 4*seq_len), dtype = int)
            for line in Lines[0:num_trees*5]:
                if counter%5 != 0 and counter!=0:
                    set_of_seqs[counter%5-1]=np.array(link(hot(line)))
                elif counter%5==0 and counter!=0:
                    self.training_data.append([rebuild(set_of_seqs, label), np.eye(3)[self.LABELS[label]]])
                    set_of_seqs=np.zeros((4, 4*seq_len), dtype=int)

                counter+=1
        np.random.shuffle(self.training_data)
        np.save(TRAINING_DATA_FOLDER,self.training_data)

if REBUILD_DATA:
    sequence_data = SequenceDataset()
    sequence_data.make_training_data()

training_data = np.load(TRAINING_DATA_FOLDER, allow_pickle = True)


RETRAIN = True



class _Model(nn.Module):
    """A neural network model to predict phylogenetic trees."""

    def __init__(self):
        """Create a neural network model."""
        super().__init__()
        self.conv = torch.nn.Sequential(
            nn.Conv1d(80, 80, 1, groups=20),
            nn.BatchNorm1d(80),
            nn.ReLU(),
            nn.Conv1d(80, 32, 1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            _ResidueModule(32),
            _ResidueModule(32),
            nn.AvgPool1d(2),
            _ResidueModule(32),
            _ResidueModule(32),
            nn.AvgPool1d(2),
            _ResidueModule(32),
            _ResidueModule(32),
            nn.AvgPool1d(2),
            _ResidueModule(32),
            _ResidueModule(32),
            nn.AdaptiveAvgPool1d(1),
        )
        self.classifier = nn.Linear(32, 3)

    def forward(self, x):
        """Predict phylogenetic trees for the given sequences.

        Parameters
        ----------
        x : torch.Tensor
            One-hot encoded sequences.

        Returns
        -------
        torch.Tensor
            The predicted adjacency trees.
        """
        x = x.view(x.size()[0], 80, -1)
        x = self.conv(x).squeeze(dim=2)
        return self.classifier(x)


class _ResidueModule(nn.Module):

    def __init__(self, channel_count):
        super().__init__()
        self.layers = torch.nn.Sequential(
            nn.Conv1d(channel_count, channel_count, 1),
            nn.BatchNorm1d(channel_count),
            nn.ReLU(),
            nn.Conv1d(channel_count, channel_count, 1),
            nn.BatchNorm1d(channel_count),
            nn.ReLU(),
        )

    def forward(self, x):
        return x + self.layers(x)



if RETRAIN:
    
    loss_vector = []
    accuracy_readings = []

    
    net = _Model().to(DEVICE)

    optimizer = optim.Adam(net.parameters(),lr=.001, weight_decay = 1e-5)
    loss_func = nn.MSELoss()
    
    X = torch.Tensor([i[0] for i in training_data]).view(-1,1,4,4*seq_len).to(DEVICE)
    y = torch.Tensor([i[1] for i in training_data]).to(DEVICE)

    VAL_PCT = .05
    val_size = int(len(X)*VAL_PCT)

    X_train = X[:-val_size]
    X_dev = X[-val_size:int(-val_size/2)]
    X_test = X[int(-val_size/2):]

    y_train = y[:-val_size]
    y_dev = y[-val_size:int(-val_size/2)]
    y_test = y[int(-val_size/2):]

    
    BATCH_SIZE = 8

    EPOCHS = 7
    
    for epoch in range(EPOCHS):
        for i in range(0,len(X_train),BATCH_SIZE):

            torch.cuda.empty_cache()
            batch_X = X_train[i:i+BATCH_SIZE].view(-1, 1, 4, 4*seq_len) 
            batch_y = y_train[i:i+BATCH_SIZE]

            batch_X.to(DEVICE), batch_y.to(DEVICE)

            net.zero_grad()
            outputs = net(batch_X)
            loss = loss_func(outputs, batch_y)
            loss.backward()
            optimizer.step()
            

        loss_vector.append(float(loss))

        correct=0
        total=0

        with torch.no_grad():
            torch.cuda.empty_cache()
            net.train(False)

            predicted_dict={0:0, 1:0, 2:0}
            actual_dict={0:0, 1:0, 2:0}
            correctly_predicted_dict = {0:0, 1:0, 2:0}

            for i in range(len(X_dev)):
                real_class = torch.argmax(y_dev[i])
                actual_dict[int(real_class)] += 1
                net_out = net(X_dev[i].view(-1, 1, 4, 4*seq_len))[0]
                predicted = torch.argmax(net_out)
                predicted_dict[int(predicted)] += 1

                if predicted == real_class:
                    correct += 1
                    correctly_predicted_dict[int(predicted)] += 1

                total+=1

        accuracy_readings.append(float(correct/total))
        print(f'Epoch:{epoch} Accuracy:{float(correct/total)}')
        print(f'Predicted:{predicted_dict} , Actual:{actual_dict}, Correctly Predicted: {correctly_predicted_dict}')
 

with torch.no_grad():
    torch.cuda.empty_cache()
    net.train(False)

    predicted_dict={0:0, 1:0, 2:0}
    actual_dict={0:0, 1:0, 2:0}
    correctly_predicted_dict = {0:0, 1:0, 2:0}
    for i in range(len(X_test)):
        real_class = torch.argmax(y_test[i])
        actual_dict[int(real_class)] += 1
        net_out = net(X_test[i].view(-1, 1, 4, 4*seq_len))[0]
        predicted = torch.argmax(net_out)
        predicted_dict[int(predicted)] += 1
        if predicted == real_class:
            correct += 1
            correctly_predicted_dict[int(predicted)] += 1
        total+=1

    accuracy_readings.append(float(correct/total))

    print(f'Epoch:{epoch} Accuracy:{float(correct/total)}')
    print(f'Predicted:{predicted_dict} , Actual:{actual_dict}, Correctly Predicted: {correctly_predicted_dict}')   

torch.cuda.empty_cache()


epochs = [x for x in range(EPOCHS+1)]
loss_vector.append(loss_vector[-1])
fig1 = plt.figure()
plt.plot(epochs, accuracy_readings, label = 'accuracy')
plt.plot(epochs, loss_vector, label = 'loss')
plt.legend()
plt.xlabel('epoch #')
plt.ylabel('accuracy/loss')
plt.title(f'Batch size:{BATCH_SIZE}. Epoch count:{EPOCHS} ')
plt.show()

