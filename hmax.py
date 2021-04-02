class DeepHMAX(nn.Module):
    def __init__(self):
        super(DeepHMAX, self).__init__()
        
        self.s1_7 = nn.Conv2d(3, 4, 7)
        self.s1_7.weight = nn.Parameter(gabor7, requires_grad=False)

        self.s1_9 = nn.Conv2d(3, 4, 9)
        self.s1_9.weight = nn.Parameter(gabor9, requires_grad=False)

        self.s1_11 = nn.Conv2d(3, 4, 11)
        self.s1_11.weight = nn.Parameter(gabor11, requires_grad=False)

        self.s1_13 = nn.Conv2d(3, 4, 13)
        self.s1_13.weight = nn.Parameter(gabor13, requires_grad=False)

        self.s1_15 = nn.Conv2d(3, 4, 15) 
        self.s1_15.weight = nn.Parameter(gabor15, requires_grad=False)

        self.s1_17 = nn.Conv2d(3, 4, 17) 
        self.s1_17.weight = nn.Parameter(gabor17, requires_grad=False)

        self.s1_19 = nn.Conv2d(3, 4, 19)
        self.s1_19.weight = nn.Parameter(gabor19, requires_grad=False)

        self.s1_21 = nn.Conv2d(3, 4, 21)
        self.s1_21.weight = nn.Parameter(gabor21, requires_grad=False)
            
        self.s1_23 = nn.Conv2d(3, 4, 23)
        self.s1_23.weight = nn.Parameter(gabor23, requires_grad=False)

        self.s1_25 = nn.Conv2d(3, 4, 25)
        self.s1_25.weight = nn.Parameter(gabor25, requires_grad=False)

        self.s1_27 = nn.Conv2d(3, 4, 27)
        self.s1_27.weight = nn.Parameter(gabor27, requires_grad=False)

        self.s1_29 = nn.Conv2d(3, 4, 29)
        self.s1_29.weight = nn.Parameter(gabor29, requires_grad=False)

        self.s1_31 = nn.Conv2d(3, 4, 31)
        self.s1_31.weight = nn.Parameter(gabor31, requires_grad=False)

        self.s1_33 = nn.Conv2d(3, 4, 33)
        self.s1_33.weight = nn.Parameter(gabor33, requires_grad=False)

        self.s1_35 = nn.Conv2d(3, 4, 35)
        self.s1_35.weight = nn.Parameter(gabor35, requires_grad=False)

        self.s1_37 = nn.Conv2d(3, 4, 37)
        self.s1_37.weight = nn.Parameter(gabor37, requires_grad=False)
        
        self.c1 = nn.Sequential(
            nn.MaxPool2d((3,3), stride=2),
            nn.Conv2d(8, 12, 3),
            nn.BatchNorm2d(12, 1e-3),
            nn.ReLU(True))
        
        self.s2 = nn.Sequential(
            nn.Conv2d(12, 16, 3), 
            nn.BatchNorm2d(16, 1e-3),
            nn.ReLU(True))
        
        self.c2 = nn.Sequential(
            nn.MaxPool2d((3,3), stride=2),
            nn.Conv2d(32, 48, 4),  
            nn.BatchNorm2d(48, 1e-3),
            nn.ReLU(True))
        
        self.s3 = nn.Sequential(
            nn.Conv2d(48, 64, 3), 
            nn.BatchNorm2d(64, 1e-3),
            nn.ReLU(True))
        
        self.s2b = nn.Sequential(
            nn.Conv2d(12, 16, 5, 2), 
            nn.BatchNorm2d(16, 1e-3),
            nn.ReLU(True),
            nn.AvgPool2d(3,2),
            nn.Conv2d(16, 24, 5, 1), 
            nn.BatchNorm2d(24, 1e-3),
            nn.ReLU(True))
        
        self.c2b = nn.Sequential(
            nn.MaxPool2d(2,1),
            nn.Conv2d(48, 64, 2, 1), 
            nn.BatchNorm2d(64, 1e-3),
            nn.ReLU(True))
        
        self.c3 = nn.Sequential(
            nn.MaxPool2d((3,3), stride=2),
            nn.Conv2d(128, 96, 5, 1),
            nn.BatchNorm2d(96, 1e-3),
            nn.ReLU(True))
        
        self.s4 = nn.Sequential(
            nn.Conv2d(704, 512, 1, 1),
            nn.BatchNorm2d(512, 1e-3),
            nn.ReLU(True),
            nn.Conv2d(512, 256, 1, 1),
            nn.BatchNorm2d(256, 1e-3),
            nn.ReLU(True),
            nn.MaxPool2d(3,2))
        
        self.norm = nn.Sequential(
            nn.BatchNorm2d(num_features=4, eps=1e-3))
        
        self.x_skip = nn.AdaptiveMaxPool2d(18)
        
        self.fc1 = nn.Sequential(
            nn.Linear(256*8*8, 4096),
            nn.BatchNorm1d(4096, 1e-3),
            nn.ReLU(True))
        
        self.fc2 = nn.Sequential(
            nn.Linear(4096, 4096),
            nn.BatchNorm1d(4096, 1e-3),
            nn.ReLU(True))

        self.fc3 = nn.Linear(4096, 1000)
        self.drop = nn.Dropout(0.5)
 
    def forward(self, x):
        # S1 to C1
        x7 = self.s1_7(x)
        x7 = torch.abs(self.norm(x7))
        x9 = self.s1_7(x)
        x9 = torch.abs(self.norm(x9))
        xa = torch.cat((x7, x9),1)
        xa = self.c1(xa)

        x11 = self.s1_7(x)
        x11 = torch.abs(self.norm(x11))
        x13 = self.s1_7(x)
        x13 = torch.abs(self.norm(x13))
        xb = torch.cat((x11, x13),1)
        xb = self.c1(xb)

        x15 = self.s1_7(x)
        x15 = torch.abs(self.norm(x15))
        x17 = self.s1_7(x)
        x17 = torch.abs(self.norm(x17))
        xc = torch.cat((x15, x17),1)
        xc = self.c1(xc)

        x19 = self.s1_7(x)
        x19 = torch.abs(self.norm(x19))
        x21 = self.s1_7(x)
        x21 = torch.abs(self.norm(x21))
        xd = torch.cat((x19, x21),1)
        xd = self.c1(xd)

        x23 = self.s1_7(x)
        x23 = torch.abs(self.norm(x23))
        x25 = self.s1_7(x)
        x25 = torch.abs(self.norm(x25))
        xe = torch.cat((x21, x23),1)
        xe = self.c1(xe)

        x27 = self.s1_7(x)
        x27 = torch.abs(self.norm(x27))
        x29 = self.s1_7(x)
        x29 = torch.abs(self.norm(x29))
        xf = torch.cat((x27, x29),1)
        xf = self.c1(xf)

        x31 = self.s1_7(x)
        x31 = torch.abs(self.norm(x31))
        x33 = self.s1_7(x)
        x33 = torch.abs(self.norm(x33))
        xg = torch.cat((x31, x33),1)
        xg = self.c1(xg)

        x35 = self.s1_7(x)
        x35 = torch.abs(self.norm(x35))
        x37 = self.s1_7(x)
        x37 = torch.abs(self.norm(x37))
        xh = torch.cat((x35, x37),1)
        xh = self.c1(xh)
        
        # C1 to S2
        xa1 = self.s2(xa)
        xb1 = self.s2(xb)
        xc1 = self.s2(xc)
        xd1 = self.s2(xd)
        xe1 = self.s2(xe)
        xf1 = self.s2(xf)
        xg1 = self.s2(xg)
        xh1 = self.s2(xh)

        # S2 to C2
        xa2 = torch.cat((xa1, xb1),1)
        xa2 = self.c2(xa2)
        xb2 = torch.cat((xc1, xd1),1)
        xb2 = self.c2(xb2)
        xc2 = torch.cat((xe1, xf1),1)
        xc2 = self.c2(xc2)
        xd2 = torch.cat((xg1, xh1),1)
        xd2 = self.c2(xd2)

        # C2 to S3
        xa2 = self.s3(xa2)
        xb2 = self.s3(xb2)
        xc2 = self.s3(xc2)
        xd2 = self.s3(xd2)

        # S3 to C3
        xa3 = torch.cat((xa2, xb2),1)
        xa3 = self.c3(xa3)
        xb3 = torch.cat((xc2, xd2),1)
        xb3 = self.c3(xb3)

        # C1 to S2b 
        xa2b = self.s2b(xa)
        xb2b = self.s2b(xb)
        xc2b = self.s2b(xc)
        xd2b = self.s2b(xd)
        xe2b = self.s2b(xe)
        xf2b = self.s2b(xf)
        xg2b = self.s2b(xg)
        xh2b = self.s2b(xh)

        # S2b to C2b
        xa2b = torch.cat((xa2b, xb2b),1)
        xa2b = self.c2b(xa2b)
        xb2b = torch.cat((xc2b, xd2b),1)
        xb2b = self.c2b(xb2b)
        xc2b = torch.cat((xe2b, xf2b),1)
        xc2b = self.c2b(xc2b)
        xd2b = torch.cat((xg2b, xh2b),1)
        xd2b = self.c2b(xd2b)

        # C2b to S4
        x_cat2 = torch.cat((xa2b, xb2b, xc2b, xd2b), 1)
        x_cat2 = self.x_skip(x_cat2)

        # Skip Connection 
        x_c2 = torch.cat((xa2, xb2, xc2, xd2), 1)
        x_skip = self.x_skip(x_c2)

        # C3, C2b, Skip to S4
        x_cat1 = torch.cat((xa3, xb3),1)
        x_cat1 = self.x_skip(x_cat1)
        x_cat = torch.cat((x_cat1, x_skip, x_cat2), 1)
        x_s4 = self.s4(x_cat)
        
        # Fully Connected Layers
        x_out = x_s4.view(-1, 256*8*8)
        x_out = self.drop(x_out)
        x_out = self.fc1(x_out)
        x_out = self.drop(x_out)
        x_out = self.fc2(x_out)
        x_out = self.fc3(x_out)
        
        return x_out      


