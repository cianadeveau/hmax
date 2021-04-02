class DeepHMAX(nn.Module):
    def __init__(self):
        super(DeepHMAX, self).__init_()

        self.v1 = nn.Sequential(
            nn.Conv2d(3, 64, 7),
            nn.BatchNorm2d(num_features=64, eps=1e-3),
            nn.ReLU(True),
            nn.MaxPool2d((3,3), stride=2)
                    )
        
        self.v2 = nn.Sequential(
            nn.Conv2d(64, 96, 3),
            nn.BatchNorm2d(96, 1e-3),
            nn.ReLU(True)
        )

        self.v4s = nn.Sequential(
            nn.Conv2d(96, 128, 3),
            nn.BatchNorm2d(128, 1e-3),
            nn.ReLU(True),
            nn.MaxPool2d((3,3), stride=2)
        )
        
        self.v4c = nn.Sequential(
            nn.Conv2d(128, 192, 4),
            nn.BatchNorm2d(192, 1e-3),
            nn.ReLU(True))
       
        self.teo_s1 = nn.Sequential(
            nn.Conv2d(192, 256, 3),
            nn.BatchNorm2d(256, 1e-3),
            nn.ReLU(True),
            nn.MaxPool2d((3,3), stride=2))
        
        self.teo_s2 = nn.Sequential(
            nn.Conv2d(96, 128, 5, 2),
            nn.BatchNorm2d(128, 1e-3),
            nn.ReLU(True),
            nn.AvgPool2d(3,2),
            nn.Conv2d(128, 192, 5, 1),
            nn.BatchNorm2d(192, 1e-3),
            nn.ReLU(True),
            nn.MaxPool2d(2,1))
        
        self.teo_c = nn.Sequential(
            nn.Conv2d(192, 256, 2, 1),
            nn.BatchNorm2d(256, 1e-3),
            nn.ReLU(True))
        
        self.te1 = nn.Sequential(
            nn.Conv2d(256, 192, 5, 1),
            nn.BatchNorm2d(192, 1e-3),
            nn.ReLU(True))
        
        self.te2 = nn.Sequential(
            nn.Conv2d(192+192+256, 512, 1, 1),
            nn.BatchNorm2d(512, 1e-3),
            nn.ReLU(True),
            nn.Conv2d(512, 256, 3, 1),
            nn.BatchNorm2d(256, 1e-3),
            nn.ReLU(True),
            nn.MaxPool2d(3,2))
        
        self.net = nn.Sequential(
            self.v1,
            self.v2)
        
        self.v4stream = nn.Sequential(
            self.v4s, 
            self.v4c)
        
        self.teo_s1stream = nn.Sequential(
            self.teo_s1,
            self.te1,
            nn.AdaptiveMaxPool2d(18))
        
        self.teo_s2stream = nn.Sequential(
            self.teo_s2, 
            self.teo_c,
            nn.AdaptiveMaxPool2d(18))
        
        self.x_skip = nn.AdaptiveMaxPool2d(18)
        
        self.fc1 = nn.Sequential(
            nn.Linear(256*7*7, 4096),
            nn.BatchNorm1d(4096, 1e-3),
            nn.ReLU(True))
        
        self.fc2 = nn.Sequential(
            nn.Linear(4096, 4096),
            nn.BatchNorm1d(4096, 1e-3),
            nn.ReLU(True))

        self.fc3 = nn.Linear(4096, 1000)
        self.drop = nn.Dropout(0.5)
    
    def forward(self, x):
        x_v1v2 = self.net(x)
        #skip 1
        x_v4c = self.v4stream(x_v1v2)
        x_teo_c = self.teo_s2stream(x_v1v2)
        
        #skip 2
        x_te1 = self.teo_s1stream(x_v4c)
        x_skip = self.x_skip(x_v4c)
       
        #join inputs
        x_cat = torch.cat((x_teo_c, x_te1, x_skip),1)
        x_te2 = self.te2(x_cat)
        x_out = x_te2.view(-1, 256*7*7)
        x_out = self.drop(x_out)
        x_out = self.fc1(x_out)
        x_out = self.drop(x_out)
        x_out = self.fc2(x_out)
        x_out = self.fc3(x_out)
        
        return x_out          
