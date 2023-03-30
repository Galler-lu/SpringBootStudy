import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets
import torch.optim as optim

if __name__ == '__main__':
    # class NetModel(torch.nn.Module):
    #     def __init__(self):
    #         super(NetModel, self).__init__()
    #         self.conv1 = torch.nn.Conv1d(4, 16, kernel_size=3, padding=1)
    #         self.conv_2 = torch.nn.Conv1d(4, 16, kernel_size=5, padding=2)
    #         self.conv1_1 = torch.nn.Conv1d(16, 32, kernel_size=3, padding=1)
    #         self.linear = torch.nn.Linear(10, 2)
    #         self.relu = torch.nn.ReLU()
    #         self.maxpooling1 = torch.nn.MaxPool1d(kernel_size=2)
    #         self.maxpooling2 = torch.nn.MaxPool1d(kernel_size=3)
    #         self.averpooling1 = torch.nn.AvgPool1d(kernel_size=2)
    #         self.averpooling2 = torch.nn.AvgPool1d(kernel_size=3)
    #
    #     def forrward(self, x):
    #         x = self.relu(self.conv1(x))
    #         x = self.relu(self.conv1(x))
    #         return self.averpooling2(x)

    # class NetModel(torch.nn.Module):
    #     def __init__(self):
    #         super(NetModel, self).__init__()
    #         self.conv1=torch.nn.Conv2d(1,16,kernel_size=3,padding=1)
    #         self.conv2=torch.nn.Conv2d(16,32,kernel_size=3,padding=1)
    #         self.relu=torch.nn.ReLU()
    #         self.averpooling=torch.nn.AvgPool2d(kernel_size=)

    # class NetModel(torch.nn.Module):
    #     def __init__(self):
    #         super(NetModel, self).__init__()
    #         self.conv1 = torch.nn.Conv1d(1, 16, kernel_size=3, padding=1)
    #         self.conv2 = torch.nn.Conv2d(16, 32, kernel_size=3, padding=1)
    #         self.relu = torch.nn.ReLU()
    #         self.avPooling = torch.nn.AvgPool1d(kernel_size=120)
    #         self.linear = torch.nn.Linear(32, 2)
    #
    #     def forward(self, x):
    #         x = self.relu(self.conv1(x))
    #         x = self.avPooling(self.relu(self.conv2(x)))
    #         return self.linear(x)
    #
    #
    # model = NetModel()
    #
    # criterion = torch.nn.CrossEntropyLoss()
    # optimizer = optim.SGD(model.parameters(), lr=0.01)

    # class NetModel(torch.nn.Module):
    #     def __init__(self):
    #         super(NetModel, self).__init__()
    #         self.conv_ghwa_1 = torch.nn.Conv1d(1, 16, kernel_size=3, padding=1)
    #         self.conv_ghwa_2 = torch.nn.Conv1d(16, 32, kernel_size=3, padding=1)
    #         self.relu = torch.nn.ReLU()
    #         self.averagePooling1 = torch.nn.AvgPool1d(6)  # 32*1
    #         self.averagePooling2 = torch.nn.AvgPool1d(9)
    #         self.averagePooling3 = torch.nn.AvgPool1d(20)
    #         self.linear = torch.nn.Linear(32, 2)
    #
    #     def forward(self, x, x_acc):
    #         x = self.relu(self.conv_ghwa_1(x))
    #         x = self.relu(self.conv_ghwa_2(x))
    #         x = self.averagePooling1(x)
    #         x_acc = self.averagePooling2(self.relu(self.conv_ghwa_1(x_acc)))
    #         x_acc = self.averagePooling3(self.relu(self.conv_ghwa_2(x_acc)))
    #
    #         out = torch.cat((x, x_acc), dim=1)
    #         return self.linear(out)
    #         # x=self.linear(x)
    #         # x_acc=self.linear(x_acc)

    class NetModel1(torch.nn.Module):
        def __init__(self):
            super(NetModel1, self).__init__()
            self.conv_ghwa_1 = torch.nn.Conv1d(2, 16, kernel_size=3, padding=1)
            self.conv_ghwa_2 = torch.nn.Conv1d(16, 32, kernel_size=3, padding=1)
            self.relu = torch.nn.ReLU()
            self.averagePooling1 = torch.nn.AvgPool1d(6)  # 32*1
            # self.averagePooling2 = torch.nn.AvgPool1d(9)
            # self.averagePooling3 = torch.nn.AvgPool1d(20)
            self.linear = torch.nn.Linear(6, 2)
            self.conv_ghwa_3 = torch.nn.Conv1d(64, 1, kernel_size=1)
            self.sigm = torch.nn.Sigmoid()

        def forward(self, x, x_acc):
            x = self.relu(self.conv_ghwa_1(x))
            x = self.relu(self.conv_ghwa_2(x))
            # x = self.averagePooling1(x)#平均池化
            # x_acc = self.averagePooling1(self.relu(self.conv_ghwa_1(x_acc)))
            # x_acc = self.averagePooling1(self.relu(self.conv_ghwa_2(x_acc)))
            x_acc = self.relu(self.conv_ghwa_1(x_acc))
            x_acc = self.relu(self.conv_ghwa_2(x_acc))
            # x_acc = self.averagePooling1(x_acc)#平均池化
            print("x为：", x, type(x), x.size())
            print("===" * 20)
            print("x_acc为：", x_acc, type(x_acc), x_acc.size())
            print("===" * 20)
            out = torch.cat((x, x_acc), dim=1)
            print("out为:", out, out.size())
            out = self.conv_ghwa_3(out)
            print("out为:", out, out.size())
            print(self.sigm(self.linear(out)))
            return self.sigm(self.linear(out))
            # x=self.linear(x)
            # x_acc=self.linear(x_acc)


    # x_data = torch.Tensor([[[1, 2, 3, 4, 5, 6]]])
    x_data = [[1, 2, 3, 4, 5, 6], [2, 3, 4, 5, 6, 7]]
    # x_data = torch.Tensor(x_data)
    # print(torch.Tensor(x_data), x_data.size())
    # print(x_data, x_data.size())
    x_data = torch.Tensor([x_data])
    print(x_data.size(), x_data)
    y_data = torch.Tensor([[[2, 3, 4, 5, 6, 7], [3, 4, 5, 6, 7, 8]]])
    # input = torch.randn(1, 1, 6)
    model = NetModel1()
    model(x_data, y_data)


    # input = torch.randn(1, 1, 5)
    # print(input,input.size())

    # class NetModel1(torch.nn.Module):
    #     def __init__(self):
    #         super(NetModel1, self).__init__()
    #         self.conv_ghwa_1 = torch.nn.Conv1d(1, 16, kernel_size=3, padding=1)
    #         self.conv_ghwa_2 = torch.nn.Conv1d(16, 32, kernel_size=3, padding=1)
    #         self.relu = torch.nn.ReLU()
    #         self.averagePooling1 = torch.nn.AvgPool1d(6)  # 32*1
    #         self.averagePooling3 = torch.nn.AvgPool1d(20)
    #         self.linear = torch.nn.Linear(6, 2)
    #         self.conv_ghwa_3 = torch.nn.Conv1d(64, 1, kernel_size=1)
    #         self.sigmoid = torch.nn.Sigmoid()
    #
    #     def forward(self, x, x_acc):
    #         x = self.relu(self.conv_ghwa_1(x))
    #         x = self.relu(self.conv_ghwa_2(x))
    #         x_acc = self.relu(self.conv_ghwa_1(x_acc))
    #         x_acc = self.averagePooling3(self.relu(self.conv_ghwa_2(x_acc)))
    #         out = torch.cat((x, x_acc), dim=1)
    #         out = self.conv_ghwa_3(out)
    #         return self.sigmoid(self.linear(out))
    #
    #
    # model = NetModel1()
    #
    # criterion = torch.nn.BCELoss()
    # optimizer = optim.SGD(model.parameters(), lr=0.01)

    # list1 = [[1, 2, 3, 4, 5, 6], [2, 3, 4, 5, 6, 7]]
    # list1 = torch.Tensor(list1)
    # print(torch.Tensor(list1), list1.size())

    class DiabetesDataset(Dataset):
        def __init__(self, data):
            # xy = np.loadtxt(filepath, delimiter=',', dtype=np.float32)
            self.len = data.shape[0]
            self.x_data = torch.from_numpy(data[:, :-1])
            self.y_data = torch.from_numpy(data[:, -1])

        def __getitem__(self, index):
            return self.x_data[index], self.y_data[index]

        def __len__(self):
            return self.len


    # dataset = DiabetesDataset(test_all)
    # train_loader = DataLoader(dataset=dataset, batch_size=32, shuffle=True, num_workers=2)
    print("===" * 20)
    x_data = [[[1, 2, 3], [2, 3, 4],[1,1,1]], [[3, 4, 5], [4, 5, 6],[2,2,2]],[[3, 4, 5], [4, 5, 6],[2,2,2]],[[3, 4, 5], [4, 5, 6],[2,2,2]]]
    x_data = torch.Tensor(x_data)
    print(x_data, x_data.size())
