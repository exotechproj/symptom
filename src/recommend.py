from recommendPosition import RecommendPosition
import sys

if __name__ == '__main__':
    if len(sys.argv) == 1:
        print('사용자 인덱스 번호를 함께 입력해주세요.(1 ~ 2400) 예) python recommend.py 5')
        exit(1)
    
    index = sys.argv[1]
    if int(index) > 2400:
        print('사용자 인덱스가 유효한 범위가 아닙니다.')
        exit(1)

    rp = RecommendPosition()
    rp.recommend(int(index))