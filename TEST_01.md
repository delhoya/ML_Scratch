
+ 1주차 강의 
  
  + K Nearest neighbors  
  
    + classification
      + 데이터가 주어졌을때 어떤 class에 속해있나? 
      + 가장 가까운 친구 찾기 나는 누구인가 - 유유상종 알고리즘 
      + K 에 따라 label이 변할수 있음   
    
    + Regression
      + 출력이 class가 아니라 실수값으로 나옴
      + X 입력축을 기준으로 K개 가까운 X 찾고 , 그 X들의 Y 값의 평균으로 출력     
    
    + variation
      + 단순하게 하지말고 거리까지 고려할 필요가 있지 않을까?  (Not just counting)
      + 거리에 따른 가중치 W(x) 적용 , 함수 디자인 
        + 1.  W(x) 함수 디자인 , 거리반비례 : 1/distance(x1,x2)
        + 2.  Exponential 함수 이용   
        + 3.  커널 리그레션 : 가우시안 분포 적용 (?!)
        + https://data-science-hi.tistory.com/64
         
    + Distance Measure
      + kNN 의 거리를 어떤식으로 정의? 
        + 공간정의에 따라 달라짐
        + 유클리안 거리 : 공간의 span에 따라 거리가 달라질수있음
        + 다른 거리 측정법? : 코사인 유사도 , 피어슨 correlation .., etc 
        + https://dive-into-ds.tistory.com/46
         

+ 2주차 강의  
  
  + Linear regression
    + W vector (w0,w1,w2....wn) 찾기   
    + X vector (X1,X2,......Xn)
    + loss function : MSE(Mean Squared Error)
  
  + how to ? : 모든 W에 대해서 편미분 진행 (variable와 같은 갯수 equations 생성)   
  + solve ? : 식을 전개한것을 matrix 에 적용 w = A-1*B 적용
  
  + w = (XtX) -1 * (XtY)  (X transpose() )
  
    + linear regression matrix form (단순선형회귀분석 / 정규방정식)
      + https://jangpiano-science.tistory.com/111
      +  https://mazdah.tistory.com/831 
    
    + kernel regression matrix (일반화) 
      + https://analysisbugs.tistory.com/163   
  
  + Another solving : Gradient descendant , Maximum likelyhood 
  
  + Model analysis: 
    + 1. Overfitting vs Generalization   
    + 2. evaluation : Training set and Test set 
    + 3. Validation set : K fold Validation + ETC  
      + https://deep-learning-study.tistory.com/623

+ 3주차 강의 
  + Minimizes the error (Gradient Descendant) vs maximizes the probability (Maximum likelyhood)
  
  + MLE : 
    +  Linear 모델을 가지고 데이터 D 가 가장 fit (maximize) 확률로 접근 
    +  weight 와 Standard Deviation 이 주어질때 데이터(D)의 X,Y 좌표 : P(x,y|w,s)
        + Noise 가 정규분포를 띈다고 가정?   
        + Y와 X는 확률 곱법칙으로 분리가능
        + X 와 w,s 는 독립이다 (Conditional proabability 의 chain rule 적용 시)  
        + 최종적으로 loss 함수형태 유도 가능 
  
  + Logisitic regression 
    + Sigmoid fuction : 나오는 값 0 ~ 1 사이 (확률과 비슷) 
    + Soft version of linear classifier (Using Exponential Fuction)  
      + 모 아니면 도 (0/1)같은 형태가 아니라 soft 한 형태의 classifier 구현가능
      +      

  
  + Naïve Bayesian
    
    + Naïve Bayesian Classifier
      + Probability with the strongest assumption on independence 
    + Formal Description of NBC
      + 
    
+ 4주차 강의 
  
  + Constrained(제한적조건) Optimization (for SVM)

+ SVM (Support vector machine) + Kernel 

+ Constrained Optimizaion (제한된 조건에서의 최적화)
  + min f(x) =?  
  + when g(x) = 0 and h(x) <= 0     
+ Lagrange Function (라그랑주 승수법)
  + F(x,a,b) = f(x) + sum(a*g(x)) + sum(b*h(x))   
  + min x / max a,b
  + a: Lagrange multiplier(equal constraint) /  b : kkt multiplier(inequal constraint)
+ 만족하는 식
  + F(x,a,b) 미분값 = 0  
  + g(x) = 0
  + b*h(x) = 0
  + h(x) <= 0  

+ 예시 :  x^2+x^2 = 1 / subject x1+x2 = 1
  + 변형 : min max (x1^2+x2^2 + a(x1+x2-1)) 
  + 제한 조건 추가 :  있을때마다 a,b,c . . . . .
  + 만족하는 식 :
    +  x1^2+x2^2 + a(x1+x2-1) 미분값 (x1,x2) 편미분 : 식 2개
    +  x1+x2-1 = 0 
    +  3개의 식 획득 
  + 2x1+a = 0 / 2x2+a = 0 / x1+1x =1
  + 정답 : x1 = x2 = 0.5 , a =-1

예시 : x^2+x^2 = 1 / subject x1+x2 = 1 , x1>=2
  + 만족 식:
    + F (미분 편미분 식 2개)
    + x1+x2-1 = 0 
    + a(-x1+2) = 0 
    + -x1+2 <= 0 
  + 정답 : KKT mulitiplier 를 이용하여 a가 0일때 , 0이 아닐때로 나뉘어서 case 1 . 2 조합
  
+ 결과 정리 : kkt multiplier 수에 따라서 최악에는 2^p 의 subproblem 이 있을 수 있음 

+ Dual form SVM
+ min x and max a ,b = max a,b and min x 로 변경가능 

+ Support Vector Machine
  + 직선을 하나 그려서 Boundary 를 지정하고 싶다. 어떤식으로 선을 그어야 할까?
  + 마진을 최대화 하는 Boundary !
  
  + 마진을 계산하는법?
    + 직선식 wx+b=0 , wx+b > 0 , wx+b < 0 (w,b 찾기)
     + wx+b =1 , wx+b = 0 ,  wx+b = -1  
     + 두 직선사이의 거리 계산 (수직인 직선에서 직선들이 만나는 점들의 거리)
       + 직선위의 한점 a,b 일때 W(a-b) = 0    
       + OX+ and OX- vector 가정 (가장 단순하게 법선 벡터와 평행한)
       + 1. |X+ - X-| = M   
       + 2.  X+ = X- + a * w (w벡터방향) (:상수) 
       + 1과 2식을 조합하면
         + |X+ - X-| = M  = a|w|
            + wx+b = 1 식에다가 x+ = x- + aw 대입 
            + w(x- + a*w) + b = 1 
            + 전개하면 wx- + a*w*w+b = 1
            + 정리하면 a = 2/w*w (wx- + b = -1 대입)
            + w*w = |w| 이므로 (거리) 
            + m = 2/|w|
   + w가 0으로 갈때 margin 이 max
   + 포인트,label에 따라서 constraint 식 추가 가능
   
    + 예시 : wx+b >1 , wx+b<-1 
    + 예시 : D = (1,1,-1) ,(2,2,+1)
      + min 0.5(w1^2+22^2) 
      + subject to (w1+w2+b+1 <=0 and -2w1-2w2-b+1<=0)
      + F = 0.5(w1^2+22^2) + a(w1+w2+b+1) + a2(-2w1-2w2-b+1)
      + 만족하는 식 :
      + x1 , x2에 대해서 각각 미분한식을 만족 + 4개의 추가 조건 획득  
        + case1 a1 = 0 a2 = 0  
        + case2 ,case3 case4 ... 에 따라서 case 별로 대입해보기 

+ Dual foam 변형
  + pass
  
+ Non-linear SVM
  + 데이터를 고차원으로 mapping      
  + equality constraint , inequality constraint 
  + data X -> O(x)
  + inner product 

+ Kernel trick
+ 커널을 이용하여 mapping = inner product 줄이기? 
  + (x1*x2+1)^3 으로 간단하게 표헌가능 !
  + 간단하게 만드는 커널 Transform 형태 (연산수 줄이기)    

+ 커널은 뭐가 될수있어?
  + Mercer's condition 만족  

6주차 - 10/4(월) 

+ 의사 결정 나무 모델
  + 장점 : IF / Then Rule 생성 가능 
  
  + 예측 나무 모델 : 데이터가 왔을때 Y 값을 예측하는 것
    + 하나의 Root 에서 끝마디 개수 X1, X2..Xn 로 나뉘어지는 사각형 영역으로 분류
  + 예측 나무 모델링 프로세스 : 끝마디 개수 R1 ... Rn  
    + 최상의 분할은 비용함수를 최소화 
    + 각 분할에 있는 y값의 평균으로 예측할때 오류 최소
  + 분할변수와 분할점은 어떻게 결정?
    + 뭘 기준으로 분리? 할거야?  / 분할점은 어떤식으로 할거야? 
    + 정답 : 모든 Case 를 다 바꿔보면서 Cost function Argmin 값 계산
  
+ 분류 나무 모델
  + Data 를 분할 공간으로 분할 
  + 분류 모델이면 새로운 data가 왔을때 분할공간으로 분류를 해줘야함
    + K개의 범주 
    + 끝노드 N개의 관측치 / 끝노드 m 에서 k 범주(클래스)의 비율 
    + 끝노드 m 으로 분류된 관측치 : k(m) 분류    
    + I(x1,x2) = X1,X2 가 Rn 지역에 있는가? 있으면 1 없으면 0 
    + k(m) = argmax f(x)   

+ 분류모델의 비용함수 
  + y 가 실제값이 아니라 범주라서 cost 가 없음
  + Misclassification rate or Gini index or Cross -entropy 
  + 분류를 얼마나 잘했는가? 

+ 분할 변수와 분할점 설정
  + 목표변수 y의 분포를 가장 잘 구분해내도록   
  + 불순도 비용함수 계산 : 엔트로피 계산가능 
    + information gain - 특정 변수 A 를 사용했을때 entropy 계산양 
    + 엔트로피(불확실성) 정보량을 많이 감소시킬수록 더 중요한 변수 A 
    + 무질서도를 감소시킴 ( 흐트러짐 감소 = for lowest entropy)
    + 모든 구간에서의 엔트로피를 다 계산한다음 min 의 구간을 고른다 
    
+ 계층형 모델의 단점
  + 중간에 에러 발생시 계속하여 에러 전파 (에러 누적)
  + 학습데이터의 미세한 변화에도 최종결과 크게 영향
  + 적은 개수의 노이즈에도 크게 영향
  + 최종 노드의 개수를 늘리면 과적합 위험이 생긴다 (Low Bias,Large Variance)
  + 엔트로피가 과하게 감소 -> 오버 피팅 발생 
  + 해결방안 : 랜덤포레스트 

+ 랜덤포레스트 
  + 앙상블모델 적용
    + 여러 모델들의 다수결 법칙 혹은 평균을 이용하여 예측 정확성 늘리기
  + 앙상블모델 조건
    + Base 모델이 서로 독립적
    + 무작위 예측을 수행하는 모델보다 성능이 좋은 경우   
  + 앙상블모델 오류율
    + Base 모델의 오류율이 무작위 모델보다 좋아야 함  
  + 핵심 아이디어
    + Diversity :여러개의 traing data 생성 (Bootstrap Aggregating) 
      + Bootstrapping : 각 데이터 셋 복원추출 : 원래 데이터의 수만큼 크기를 갖도록 샘플링  
      + 여러개의 개별 데아터셋 (ootstrapping)생성
      + 이론적으로 한 개체가 부트스트랩에 한번도 선택 안될 확류 : p = (1-n)^1/n = e-1 = 0.368
      + Bagging : I 함수 (0or1) 의 값 계산 -> argmax 값 선택 (ex Y-0 ,값4 or y = 1 ,값 6 : 1선택)
      + Training Accuracy 가중치 적용하여 Bagging 개념에 합산
    + Random subspace : 변수 무작위로 선택 
      + 모델 구성에 활용할 변수 무작위로 선택 
      + 무작위로 선택해가며 full grown tree 가 되도록 구성
      + 원래 변수의 수보다 적은 변수를 임의로 선택하여 해당 변수의 분기점 고려 
+ 에러
  + 개별의 Tree 는 과적합 될수 있음
  + 트리수가 충분히 많을때 수렴됨
    + Generalization Error : lo * (1-s^2) / s^2 : 작을수록 좋음 
    + lo : 트리 사이에 평균 상관관계  
    + s : 올바르게 예측한 tree 와 잘못 예측한 tree 차이의 평균 
    + s : 개별트리 정확도가 높을수록 증가
    + lo : Bagging 과 random subspace 기법은 무작위성 최대화 (독립성,일반화,무작위성) 

+ 중요 변수 선택


+ 7주차 
+ Clustering 
  
  + Kmeans
    + Unsupervised 
  
  + Gaussian Mixture
    + Randomly choose one of Gaussians, and generate a data
  
+ 8주차
  + PCA
    + Covariance Matrix : 대칭행렬 (Symmetric matrix) 
      + 데이터의 분포 유추 가능  
      + Diagonal : 원형 (모든 영향이 같음 = 원)
      + Diagonal X  : 데이터 형태가 주축에 영향을 가장 많이 받은 치우친 형태  
  + SVD 
    + 고유값 분해 : n x n 정방행렬 A가 n개의 일차독립인 고유벡터 가질 때 고유값으로 분해
      + 대칭행렬 : 항상 고유값 대각화가 가능 + 직교행렬(orthogonal matrix) 대각화가 가능
    + 특이값 분해 : m x n 크기의 데이터 행렬 A를 분해
      + https://techblog-history-younghunjo1.tistory.com/66
  + PCA 
      + 입력 데이터들의 공분산 행렬(covariance matrix)에 대한 고유값 분해(eigendecomposition)
        + 공분산행렬 : 입력 데이터가 얼마만큼이나 함께 변하는가 : 모든 값의 내적 연산을 취함 (벡터내적: 유사도판단) 
        + 자신의 내적을 위한 전치행렬 (Transpose()) 내적 
      + C = 1/n * X * Xtranspose() 
        + C(공분산행렬) 의 고유값 분해 진행 S*A*S_transpose()
        + 고유벡터 : 주성분 벡터 (분포에서 분산이 큰 방향)
        + 고유값 : 분산의 크기
      
  + HMM 
    + Class A and Class B : 2개의 Sequence 가 있을 때 사용하는 모델 
      
      + 주어진 String 은 어느 Class에 속한다고 봐야할까?  
      + Supervised learning (Sequence) /  Class Classification 
      + State 가 주어진 Machine 가정
        + 초기 A에 있을 확률 : Pi A 
        + 초기 B에 있을 확률:  Pi B
        + 다음 A -> A or A - > B : 
        + 다음 B -> B or B - > A :  
      
      + Markov Model : T+1 시점은 T 시점 영향
        + (Pi A) * P(A2|A1) * P (A3|A2) * ... 표현가능
        
      + 1. 모델이 주어졌을때 Output Sequence 확률 ? = forward로 곱연산 
      + 2. 주어진 모델과 관측에 대해서 State 의 Path? = Highest Prob. (maximize)
      + 3. 데이터가 주어졌을때(O) 가장 잘되는 모델? = Expectation and Maximize 
      
        + Evaluation , (신호가 어디에서 왔는가?) : 
        + Decoding , (Word 에서 형태소 POS 분석하는것)  : 
        + Learning or estimation  (A,B Pi 를 어떤식으로 학습을 하는가?)
 
   + A,B,Pi 
    + 1. Forward & Backward
      + 
    + 2. Viterbi 알고리즘
      + 
    + 3. Baul Welch 알고리즘 
      + 