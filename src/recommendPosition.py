import pandas as pd
import time
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
import re


class RecommendPosition():
    def __init__(self):
        # 데이터 불러오기
        self.whole_data = pd.read_csv('position_data.csv')        
        self.whole_data.fillna('', inplace=True)
        self.whole_data_parse = self.whole_data.copy()
        print('데이터 로딩 시작')
        print(f'전체 데이터 건수 : {self.whole_data.shape}')

        self.row_count_dict = self.split_position()
        print(f'총 직무 갯수 : {len(self.row_count_dict)}')

        # 자격증 정보 파싱
        self.whole_data_parse['parsing_certification'] = self.whole_data['base_certification'].fillna('').apply(self.base_certification)

        # 직무 정보 파싱
        self.whole_data_parse['parsing_work'] = self.whole_data['base_hopework'].fillna('').apply(self.make_hopework)

        # 기술 정보 파싱
        self.whole_data_parse['parsing_skill'] =  self.whole_data['base_skill'].fillna('').apply(self.make_skill)

        # 불필요한 컬럼 삭제
        self.delete_columns()

        self.whole_data_arr = self.whole_data_parse.to_numpy()
        self.X_train, self.X_test = train_test_split(self.whole_data_arr, test_size=0.1)
        self.corpus = self.make_corpus()

        self.tfidf = TfidfVectorizer()
        self.tfidf_m = self.tfidf.fit_transform(self.corpus)

        # 보직 index and name
        self.index_to_bogic = self.make_bosic()
        print('데이터 로딩 완료')
        print()
        

    def make_bosic(self):
        '''
        인덱스 번호를 통해 보직명을 반환한다.
        '''
        # index to 보직
        index_to_bogic = {}
        for index, i in enumerate(self.row_count_dict):
                index_to_bogic[index] = i    
        
        return index_to_bogic

    def make_corpus(self):
        '''
        corpus 생성
        '''
        #whole_data_arr = self.whole_data_parse.to_numpy()

        # 웹개발자 [[], [], [], [], []]
        # 보직별로 자격증, 스킬, 언어, 성별, 나이등을 취합한다.
        for k, v in self.row_count_dict.items():
            for i in self.X_train:
                if k in i[1]:
                    if len(i[0]) > 0 and i[0][0] != '':
                        v[0].extend(i[0])
                    if len(i[2]) > 0 and i[2][0] != '':
                        v[1].extend(i[2])
                    if i[3] != '':
                        v[2].append(i[3])
                    if i[4] != '':
                        v[3].append(i[4])
                    if len(i[5]) > 0 and i[5][0] != '':
                        v[4].append(i[5])

        c_dict = {}
        
        for k, v in self.row_count_dict.items():
            value_list = [[],[],[],[],[],[]]
            tmp_dict = dict(sorted(Counter(v[0]).items(), key=lambda item: -item[1]))
            
            value_list[1] = {i:v for i,v in tmp_dict.items() if v >= 10}
            for index, i in enumerate(value_list[1]):
                if index < 5:
                    for _ in range(10): value_list[0].append(i)
                    
            tmp_dict = dict(sorted(Counter(v[1]).items(), key=lambda item: -item[1]))
            value_list[2] = {i:v for i,v in tmp_dict.items() if v >= 10}
            for index, i in enumerate(value_list[2]):
                if index < 10:
                    for _ in range(10): value_list[0].append(i)
                    
            tmp_dict = dict(sorted(Counter(v[2]).items(), key=lambda item: -item[1]))
            value_list[3] = {i:v for i,v in tmp_dict.items() if v >= 10}
            for index, i in enumerate(value_list[3]):
                value_list[0].append(i)
                    
            tmp_dict = dict(sorted(Counter(v[3]).items(), key=lambda item: -item[1]))
            value_list[4] = {i:v for i,v in tmp_dict.items() if v >= 10}
            for index, i in enumerate(value_list[4]):
                value_list[0].append(i)
                    
            tmp_dict = dict(sorted(Counter(v[4]).items(), key=lambda item: -item[1]))
            value_list[5] = {i:v for i,v in tmp_dict.items() if v >= 10}
            for index, i in enumerate(value_list[5]):
                value_list[0].append(i)
            
            value_list[0] = value_list[0]
            c_dict[k] = value_list

            # 단어 합 생성
            corpus = []
            for k, v in c_dict.items():
                corpus.append(' '.join(v[0]))

        return corpus


    def delete_columns(self):
        self.whole_data_parse = self.whole_data_parse.drop(['base_education', 'base_skill',
       'base_corecompetency', 'base_career', 'base_intern', 'base_learn',
       'base_certification', 'base_experience', 'base_language',
       'base_introduction', 'base_portfolio', 'base_hopework', 'base_awards',
       'base_personality-test'
       ], axis=1)
        #whole_data_parse

        whole_data_parse_tmp = pd.DataFrame()
        whole_data_parse_tmp['parsing_certification'] = self.whole_data_parse['parsing_certification']
        whole_data_parse_tmp['parsing_work'] = self.whole_data_parse['parsing_work']
        whole_data_parse_tmp['parsing_skill'] = self.whole_data_parse['parsing_skill']
        whole_data_parse_tmp['parse_age'] = self.whole_data_parse['parse_age']
        whole_data_parse_tmp['parse_gender'] = self.whole_data_parse['parse_gender']
        whole_data_parse_tmp['parse_language'] = self.whole_data_parse['parse_language']

        self.whole_data_parse = whole_data_parse_tmp


    def split_position(self):
        # 직무분리
        base_hopework = self.whole_data['base_hopework'].fillna('')
        base_hopework_list = []
        for i in base_hopework:
            if len(i.split('직무/')) > 1:
                hopework = i.split('직무/')[1].split('/산업')[0].split('/')
                for k in hopework:
                    if ',' in k:
                        base_hopework_list.extend(k.replace(' ', '').split(','))
                    else:
                        base_hopework_list.append(k)

        # 중복 카운트가 많은 순으로 정렬
        count_dict = dict(sorted(Counter(base_hopework_list).items(), key=lambda x: -x[1]))

        # 직무 카운트가 10개 이상인 것들만 추출
        row_count_dict = {k: [[], [], [], [], []] for k, v in count_dict.items() if v > 10}
        
        return row_count_dict
    
    
    def base_certification(self, certification):
        '''
        자격증 파싱
        '''
        licenses_list = []
        
        # 불필요 워드 제거
        certification = re.sub('(1급|1종|2급|2종|3급|3종|\(국가공인\))', '', certification)
        # 2020. 11/산업안전기사 한국산업인력공단/2020. 08/초경량비행장치 조종자
        licenses_list.extend(certification.split('/')[1::2]) 

        license = []
        for licenses in licenses_list:
            s = licenses.split(' ')
            #print(s)
            if (len(s) == 1 or len(s) == 2) and licenses.strip() != '':        
                license.append(s[0].strip())
            elif len(s) == 3:
                c = (s[0] + s[1]).strip()
                if c != '':
                    license.append(c)
            elif len(s) >= 4 and s[0].strip() != '':
                license.append(s[0].strip())

        license_set = set(license)
        license_final = []
        for i in license_set:
            if i.find('운전') != -1:
                if i.find('자동차운전') != -1 or i.find('보통운전면허') != -1 or i.find('운전면허보통') != -1 or i.find('운전면허증') != -1:
                    license_final.append('자동차운전면허')
                elif i.find('지게차운전') != -1:
                    license_final.append('지게차운전기능사')
                elif i.find('택시') != -1:
                    license_final.append('택시운전자격')
                elif i.find('버스') != -1:
                    license_final.append('버스운전자격')
                elif i.find('특수') != -1:
                    license_final.append('특수운전면허')
                elif i.find('굴삭기') != -1:
                    license_final.append('굴삭기운전기능사')
            else:
                license_final.append(i)

        return license_final
    

    def make_hopework(self, base_hopework):
        '''
        직무정보 파싱
        '''
        base_hopework_list = []
        if len(base_hopework.split('직무/')) > 1:
            hopework = base_hopework.split('직무/')[1].split('/산업')[0].split('/')
            for k in hopework:
                if ',' in k:
                    base_hopework_list.extend(k.replace(' ', '').split(','))
                else:
                    base_hopework_list.append(k)
        
        return base_hopework_list


    def make_skill(self, skills):
        '''
        기술정보 파싱
        '''
        result = [i.lower() for i in skills.replace(' ', '').split('/')]
        return result

    
    def accuracy(self, f_index, input_data):
        test_data = []
        
        test_data.extend(input_data[f_index][0])
        test_data.extend(input_data[f_index][2])
        test_data.append(input_data[f_index][3])
        test_data.append(input_data[f_index][4])
        test_data.append(input_data[f_index][5])
        
        tfidf_m2 = self.tfidf.transform([' '.join(test_data)])

        sim2 = cosine_similarity(self.tfidf_m, tfidf_m2)
        #print(sim2)
        sim_score_ = list(enumerate(sim2.flatten()))
        #print(sim_score_)
        sim_score2_ = sorted(sim_score_, key=lambda x: x[1], reverse=True)
        #print(sim_score2_)
        # 정렬된 데이터의 인텍스 구하기
        bogic_idx_ = [idx[0] for idx in sim_score2_[:5]]
        
        target = input_data[f_index][1]
        
        # 인텍스를 이용하여 보직이름 구하기
        result_arr = []
        success = False
        for index, i in enumerate(bogic_idx_):
            #print(index, i)
            result_arr.append(self.index_to_bogic[i])

        for i in target:
            if i in result_arr:
                success = True

        print(target, '=>', result_arr, success)

        return success
    

    def accuracy_sum(self):
        total = 0
        failed = 0
        result_index = []
        full_data_cnt = len(self.whole_data_arr)
        train_data = self.X_train
        test_data = self.X_test

        for i in range(len(test_data)):
            acc = self.accuracy(i, test_data)
            if acc == True:
                result_index.append(i)    
                total += 1
            else:
                failed += 1

        print('=====================================')
        print(f'전체사용자 수 : {full_data_cnt}')
        print('=====================================')
        print(f'훈련셋 데이터 수 : {len(train_data)}')
        print(f'테스트셋 데이터 수 : {len(test_data)}')
        print('-------------------------------------')
        print(f'직무를 성공적으로 예측하지 못한 수 : {failed}')
        print(f'정확도( (테스트셋 데이터 수 - 예측실패 수 ) / 테스트셋 데이터 수) : {((len(test_data) - failed)) / len(test_data):%}')
    

    def recommend(self, f_index):
        start = time.time()
        test_data = []
        #print(self.whole_data_parse.iloc[f_index]['parsing_work'])
        #print(self.whole_data_parse.iloc[f_index]['parsing_certification'])
        #print(self.whole_data_parse.iloc[f_index]['parsing_skill'])
        #print(self.whole_data_parse.iloc[f_index]['parse_age'])
        #print(self.whole_data_parse.iloc[f_index]['parse_gender'])
        #print(self.whole_data_parse.iloc[f_index]['parse_language'])

        target = self.X_test[f_index][1]

        test_data.extend(self.X_test[f_index][0])
        test_data.extend(self.X_test[f_index][2])
        test_data.append(self.X_test[f_index][3])
        test_data.append(self.X_test[f_index][4])
        test_data.append(self.X_test[f_index][5])

        #test_data.extend(['정보처리기사', '자동차운전면허', '워드프로세서', '정보처리산업기사'])
        #test_data.extend(['java', 'html', 'css', 'mybatis', 'oracle', 'mariadb', 'spring', 'ajax'])
        #test_data.append('20대')
        #test_data.append('남자')
        #test_data.append('영어')
        
        tfidf_m2 = self.tfidf.transform([' '.join(test_data)])

        index_to_bogic = self.make_bosic()

        sim2 = cosine_similarity(self.tfidf_m, tfidf_m2)
        
        sim_score_ = list(enumerate(sim2.flatten()))
        
        sim_score2_ = sorted(sim_score_, key=lambda x: x[1], reverse=True)
        
        # 정렬된 데이터의 인텍스 구하기
        bogic_idx_ = [idx[0] for idx in sim_score2_[:5]]
        result_arr = []

        print(f'희망보직 : {target}')
        print(f'추천보직 :')
        # 인텍스를 이용하여 보직이름 구하기
        for index, i in enumerate(bogic_idx_):
            #print(index_to_bogic[i], sim_score2_[index][1])
            result_arr.append(self.index_to_bogic[i])
            print((index + 1), index_to_bogic[i])

        success = False
        for i in target:
            if i in result_arr:
                success = True
        
        print(f'추천성공여부 : {success}')

        end = time.time()
        print(f'소요시간 : {end - start:.5f} sec')

    def recommend_from_data(self, cetification: list, skill: list, gender: str, language: str, age: str):
        
        test_data = []
        test_data.extend(cetification)
        test_data.extend(skill)
        test_data.append(age)
        test_data.append(gender)
        test_data.append(language)
        
        tfidf_m2 = self.tfidf.transform([' '.join(test_data)])

        index_to_bogic = self.make_bosic()

        sim2 = cosine_similarity(self.tfidf_m, tfidf_m2)
        
        sim_score_ = list(enumerate(sim2.flatten()))
        
        sim_score2_ = sorted(sim_score_, key=lambda x: x[1], reverse=True)
        
        # 정렬된 데이터의 인텍스 구하기
        bogic_idx_ = [idx[0] for idx in sim_score2_[:5]]
        result_arr = []

        # 인텍스를 이용하여 보직이름 구하기
        for index, i in enumerate(bogic_idx_):
            #print(index_to_bogic[i], sim_score2_[index][1])
            result_arr.append(self.index_to_bogic[i])

        return result_arr