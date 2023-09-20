
from symptom import Symptom
import sys


if __name__ == '__main__':
    if len(sys.argv) == 1:
        print('사용자 인덱스 번호를 함께 입력해주세요.(1 ~ 400) 예) python symptom_test.py 5')
        exit(1)
    
    index = sys.argv[1]
    if int(index) > 400:
        print('사용자 인덱스가 유효한 범위가 아닙니다.')
        exit(1)

    symptom = Symptom()
    symptom.load_model()
    symptom.accuracy_test(int(index))