def random_update_dataset(possible_selection, update_size, train_set, cifar3, trainloader):
    random_sample_batch= random.sample(possible_selection, update_size)
    for item in random_sample_batch:
        train_set.update(trainset.data[cifar3[item]], trainset.targets[cifar3[item]])
    trainloader = torch.utils.data.DataLoader(train_set, batch_size=100, shuffle=True, num_workers=0)
    return random_sample_batch
    