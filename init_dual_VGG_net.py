def init_dual_vgg16_network(num_output):
    net = vgg16(pretrained=False)
    net2 = vgg16(pretrained=False)
    num_features = net.output.in_features
    num_features2 = net2.output.in_features
    features = list(net.output.children())[:-1]
    features2 = list(net2.output.children())[:-1]
    features.extend([nn.Linear(num_features, num_output)])
    features2.extend([nn.Linear(num_features2, num_output)])
    net.output = nn.Sequential(*features)
    net2.output = nn.Sequential(*features2)
    return net, net2