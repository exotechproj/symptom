import pandas as pd
import numpy as np
import warnings
import torch
import torch.nn as nn
import numpy as np
import random
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from pickle import dump, load
import time

class CustomModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Linear(49, 100),
            nn.ReLU(),
            nn.Linear(100, 200),
            nn.ReLU(),
            nn.Linear(200, 200),
            nn.ReLU(),
            nn.Linear(200, 50),
            nn.ReLU(),
            nn.Linear(50, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        x = self.layer(x)
        return x
    

class CustomDataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.len = len(x)
        #print(self.len)
        
    def __getitem__(self, index):
        x = torch.FloatTensor(self.x[index])
        y = torch.FloatTensor(self.y[index])
        return x, y
    
    def __len__(self):
        return self.len
    

class Symptom:
    def __init__(self):
        print('--------------------------------')
        print('데이터 불러오기')
        print('--------------------------------')
        self.data = pd.read_csv('eda2.csv').fillna(0)

        self.device = self.init_device()

        dumy_dict = {1:'a', 2:'b', 3:'c', 4:'d', 5:'e'}

        data2 = self.data.replace({'감성지수':dumy_dict})
        data2.replace({'만족도':dumy_dict}, inplace=True)
        data2.replace({'업무성과점수':dumy_dict}, inplace=True)
        data2.replace({'근무태도':dumy_dict}, inplace=True)

        duty_dict = {1:'없음', 0:'있음'}
        data2.replace({'DUTY':duty_dict}, inplace=True)

        data2['NAI'] = data2['NAI'].apply(self.nai_group)
        data2['WORKING_YEAR'] = data2['WORKING_YEAR'].apply(self.work_group)     

        self.label = data2['EMPLOYEE_STATUS'].to_numpy().reshape(-1, 1)

        data2 = data2.drop(['EMPLOYEE_STATUS'], axis=1)

        data2 = pd.get_dummies(data2)

        data2 = data2.to_numpy()
        self.data2 = data2
        print(f'데이터 총 건수 : {len(data2)}')
        #print(data.shape, self.label.shape)

        #self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(data, self.label, test_size=0.2, random_state=61, stratify=self.label )
        #self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(self.X_train, self.y_train, test_size=0.2, random_state=61, stratify=self.y_train )
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(data2, self.label, test_size=0.2, stratify=self.label )
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(self.X_train, self.y_train, test_size=0.2, stratify=self.y_train )
        print('데이터 shape : ', self.X_train.shape, self.y_train.shape, self.X_val.shape, self.y_val.shape, self.X_test.shape, self.y_test.shape)
        print()

        self.train_dataset = CustomDataset(self.X_train, self.y_train)
        self.val_dataset = CustomDataset(self.X_val, self.y_val)

        self.train_dataloader = DataLoader(self.train_dataset, batch_size=8, shuffle=True)
        self.val_dataloader = DataLoader(self.val_dataset, batch_size=8, shuffle=False)

        self.model = CustomModel().to(self.device)
        self.criterion = torch.nn.BCELoss().to(self.device)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.001)

    def nai_group(self, values):
        if values <= 25:
            return 'a'
        elif values <= 30:
            return 'b'
        elif values <= 35:
            return 'c'
        elif values <= 40:
            return 'd'
        else:
            return 'e'

    def work_group(self, values):
        if values <= 0:
            return 'a'
        elif values <= 1:
            return 'b'
        elif values <= 3:
            return 'c'
        elif values <= 5:
            return 'd'
        elif values <= 7:
            return 'e'
        elif values <= 9:
            return 'f'
        else:
            return 'g'
        

    def init_device(self):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        #torch.manual_seed(777)
        #if device == 'cuda':
        #    torch.cuda.manual_seed_all(777)
        #print(device)
        return device
    

    def train(self):
        print('Train 시작')
        TRAIN_LOSS = []
        VAL_LOSS = []
        best_loss = 10000000
        patience_limit = 3
        patience_check = 0

        for epoch in range(500):
            train_loss = 0
            val_loss = 0
            
            self.model.train()
            for x, y in self.train_dataloader:
                self.optimizer.zero_grad()
                
                x = x.to(self.device)
                y = y.to(self.device)
                
                output = self.model(x)
                #print(output.size())
                #print('___')
                #print(y.size())
                loss = self.criterion(output, y)
                train_loss += loss.item()
                loss.backward()
                self.optimizer.step()
                
            train_loss = train_loss / len(self.train_dataloader)
            TRAIN_LOSS.append(train_loss)
            
            self.model.eval()
            for x, y in self.val_dataloader:
                x = x.to(self.device)
                y = y.to(self.device)
                
                output = self.model(x)
                loss = self.criterion(output, y)
                val_loss += loss.item()
                
            val_loss = val_loss / len(self.val_dataloader)
            VAL_LOSS.append(val_loss)
            print(f'epoch : {epoch + 1}, train_loss={train_loss}, val_loss={val_loss}')
            
            if val_loss > best_loss:
                patience_check += 1
                
                if patience_check >= patience_limit:
                    break
            else:
                best_loss = val_loss
                patience_check = 0
            
        torch.save(self.model.state_dict(), 'model_symptoms_test.pt')
        print('Train 종료')
        print()

        self.model.eval()
        with torch.no_grad():
            hypothesis = self.model(torch.FloatTensor(self.X_test).to(self.device))
            pred = [1 if a > 0.5 else 0 for a in hypothesis.cpu().numpy()]
            y_test = self.y_test.reshape(-1)
            a = sum(pred == y_test)/len(y_test)
            print(f'정확도 테스트 : {a:%}')

    def load_model(self):
        '''
        이미 학습된 모델 로드
        '''
        self.loaded_model = CustomModel().to(self.device)
        self.loaded_model.load_state_dict(torch.load('model_symptoms.pt'))
    
    def accuracy(self):
        '''
        테스트 데이터를 이용한 정확도 측정
        '''
        self.loaded_model.eval()
        with torch.no_grad():
            hypothesis = self.loaded_model(torch.FloatTensor(self.X_test).to(self.device))
            pred = [1 if a > 0.5 else 0 for a in hypothesis.cpu().numpy()]
            y_test = self.y_test.reshape(-1)
            #a = sum(pred == y_test)/len(y_test)
            #print(f'정확도 : {a:.5%}')
            failed = 0
            for i, r in zip(pred, y_test):
                if i != r:
                    failed += 1
            print(f'전체 사용자 수 : {len(y_test)}')
            print(f'이상징후를 성공적으로 예측하지 못한 사용자의 수 : {failed}')
            print(f'예측정확도 : {( (len(y_test) - failed) / (len(y_test)) ):.5%}')


    def accuracy_test(self, idx):
        '''
        테스트 데이터를 이용한 정확도 측정2
        '''
        start = time.time()
        
        print('--------------------------------')
        print(f'사용자 정보 : {idx}')
        print('--------------------------------')
        employee = self.data.iloc[idx]
        print(f'직급 : {employee["POSITION"]}')
        print(f'직책여부 : {employee["DUTY"]}')
        print(f'근무년수 : {employee["WORKING_YEAR"]}')
        print(f'나이 : {employee["NAI"]}')
        print(f'기술등급 : {employee["TECH_LEVEL"]}')
        print(f'업무구분 : {employee["WORKTYPE"]}')
        print(f'감성지수 : {employee["감성지수"]}')
        print(f'업무만족도 : {employee["만족도"]}')
        print(f'업무성과점수 : {employee["업무성과점수"]}')
        print(f'근무태도 : {employee["근무태도"]}')
        print(f'이상징후 : {employee["EMPLOYEE_STATUS"]}')
        print('--------------------------------')

        self.loaded_model.eval()
        with torch.no_grad():
            hypothesis = self.loaded_model(torch.FloatTensor(self.data2[idx]).to(self.device))
            pred = [1 if a > 0.5 else 0 for a in hypothesis.cpu().numpy()]
            print(f'이상징후 여부 : {pred}')
        
        end = time.time()
        print(f'소요시간 : {end - start:.5f} sec')


    def accuracy_test(self, idx):
        '''
        테스트 데이터를 이용한 정확도 측정2
        '''
        start = time.time()
        
        print('--------------------------------')
        print(f'사용자 정보 : {idx}')
        print('--------------------------------')
        employee = self.data.iloc[idx]
        print(f'직급 : {employee["POSITION"]}')
        print(f'직책여부 : {employee["DUTY"]}')
        print(f'근무년수 : {employee["WORKING_YEAR"]}')
        print(f'나이 : {employee["NAI"]}')
        print(f'기술등급 : {employee["TECH_LEVEL"]}')
        print(f'업무구분 : {employee["WORKTYPE"]}')
        print(f'감성지수 : {employee["감성지수"]}')
        print(f'업무만족도 : {employee["만족도"]}')
        print(f'업무성과점수 : {employee["업무성과점수"]}')
        print(f'근무태도 : {employee["근무태도"]}')
        print(f'이상징후 : {employee["EMPLOYEE_STATUS"]}')
        print('--------------------------------')

        self.loaded_model.eval()
        with torch.no_grad():
            hypothesis = self.loaded_model(torch.FloatTensor(self.data2[idx]).to(self.device))
            pred = [1 if a > 0.5 else 0 for a in hypothesis.cpu().numpy()]
            print(f'이상징후 여부 : {pred}', hypothesis.cpu().numpy())
        
        end = time.time()
        print(f'소요시간 : {end - start:.5f} sec')        


    def symptom_score(self, data):
        #print(data)
        #output1 = list(self.data2[0])
        #print(pd.get_dummies(df))
        #print('------------')
        _pos_arr = ['담당', '사원', '상무이사', '수석', '이사', '인턴', '전무이사', '책임']
        _duty_arr = [1, 0]
        _work_year = ['a', 'b', 'c', 'd', 'e', 'f', 'g']
        _age = ['a', 'b', 'c', 'd', 'e']
        _tech = ['고급', '중급', '초급', '특급']
        _worktype = ['개발직', '경영관리직', '연구직']
        _emo = [1, 2, 3, 4, 5]

        output = [[], [], [], [], [], [], [], [], [], []]
        for i in _pos_arr:
            if i == data[0]: output[0].append(1)
            else: output[0].append(0)
        
        for i in _duty_arr:
            if i == data[1]: output[1].append(1)
            else: output[1].append(0)
        
        for i in _work_year:
            if i == self.work_group(data[2]): output[2].append(1)
            else: output[2].append(0)
        
        for i in _age:
            if i == self.nai_group(data[3]): output[3].append(1)
            else: output[3].append(0)
        
        for i in _tech:
            if i == data[4]: output[4].append(1)
            else: output[4].append(0)
        
        for i in _worktype:
            if i == data[5]: output[5].append(1)
            else: output[5].append(0)
        
        for i in _emo:
            if i == data[6]: output[6].append(1)
            else: output[6].append(0)
        
        for i in _emo:
            if i == data[7]: output[7].append(1)
            else: output[7].append(0)
        
        for i in _emo:
            if i == data[8]: output[8].append(1)
            else: output[8].append(0)
        
        for i in _emo:
            if i == data[9]: output[9].append(1)
            else: output[9].append(0)
        
        output2 = []

        for i in range(10):
            output2.extend(output[i])

        #for i, k in zip(output1, output2):
        #    print(i == k)

        #print(len(output1), len(output2))
        #print(np.array(output2))


        self.loaded_model.eval()
        with torch.no_grad():
            hypothesis = self.loaded_model(torch.FloatTensor(output2).to(self.device))
            pred = hypothesis.cpu().numpy()
            #print(f'이상징후 여부 : {pred}')
        
        return pred

        
        
        





        

        
            
    
