import torch
import torch.nn.functional as F
import torch.nn as nn
import copy 
criterion = nn.CrossEntropyLoss()
def train(epoch, model_net, model_optimizer,trainloader, device, if_print = False):
    if if_print:
        print('\nEpoch: %d' % epoch)
    train_loss = 0
    correct = 0
    total = 0
    model_net.train()
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        model_optimizer.zero_grad()
        outputs = model_net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        model_optimizer.step()      
        if batch_idx % 10 == 0 and if_print == True:
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += predicted.eq(targets.data).cpu().sum()
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(inputs), len(trainloader.dataset),
                100. * batch_idx / len(trainloader), loss.item()))
            
def test_test(epoch, model_net, testloader, device):
    model_net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model_net(inputs)
            loss = criterion(outputs, targets)
            test_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += predicted.eq(targets.data).cpu().sum()
            
    acc = 100.*correct/total
    best_acc = acc
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(testloader.dataset),
        100. * correct / len(testloader.dataset)))      
    return (100* float(correct)/float(len(testloader.dataset)))

def test_val( model_net, best_net, val_loader, best_score, device):
    model_net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(val_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model_net(inputs)
            loss = criterion(outputs, targets)
            test_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += predicted.eq(targets.data).cpu().sum()

    # Save chseckpoint.
    acc = float(100*correct.item()/total)
    best_acc = acc
    if acc > best_score:
        best_score = acc
        best_net = copy.deepcopy(model_net.state_dict())   
    return best_score, best_net

def test_acc_per_class(model_net, testloader, device):
    class_id = [i for i in range(10)]
    class_correct = [0 for i in range(10)]
    class_count = [0 for i in range(10)]
    model_net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model_net(inputs)
            loss = criterion(outputs, targets)
            test_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            for i in range(len(predicted)):
                for index in range(len(class_id)):
                    if class_id[index] == targets[i]:
                        class_count[index] += 1
                        if predicted[i].eq(targets[i].data).cpu():
                            class_correct[index] += 1
    return (class_count,class_correct)    

def apply_dropout(m):
    if type(m) == nn.Dropout:
        m.train()
        


        