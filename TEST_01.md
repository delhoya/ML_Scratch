+ 기초 행렬 연산 
  + https://ratsgo.github.io/linear%20algebra/2017/03/14/operations/


+ 1주차 강의 
  
  + K Nearest neighbors : 
    + 특징
      + Distance 기반 (Instance-based learning) 
      + Instance-based : 관측치만을 이용하여 새로운 데이터 예측
      + Memory-based : 모든 학습데이터 메모리에 저장
      + Lazy learning : 모델 학습 X 
    
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
         
    + 생각 해볼 것
      + Which k is better?
        + Small k : higher variance (less stable)
        + Large k : higher bias (less precise)   
      
      + Proper choice of k
        + Depending on the data
        + Use Cross-validation
        
+ 2주차 강의  
  
  + Regression Model :
    + 회귀 모델에서 우리가 추정해야 하는 미지수는 독립 변수나 종속 변수가 아니라 회귀 계수 
    + 회귀 모델은 모델링 대상을 회귀 계수의 선형 결합만으로 표현할 것인지 여부에 따라 ‘선형’ 회귀 모델과 ‘비선형’ 회귀 모델로 구분됩니다.
    + https://danbi-ncsoft.github.io/study/2018/05/04/study-regression_model_summary.html
    
  + Linear regression
  
    + 다중 선형회귀 : 여러개 독립변수 (X1,...,Xn) 다른 가중치(W1,...Wn) , 여러개의 종속변수 (Y1,...,Yn)
    + http://piramvill2.org/?p=3224&
  
  + how to ? : 모든 W1...Wn에 대해서 편미분 진행 
    + variable와 같은 갯수 equations 생성   
  + solve ? : 
    
    + Y : 반응변수 벡터 (n*1) 
    + X : 설명변수 행렬 (n * (p+1)) 
    + e : 잔차 벡터 (n*1)
    
    + + B : 회귀계수 벡터 ((p+1) * 1)
    
  + Argmin |F(x)-y|    
      + F(X) = W0 + W1X1 + W2X2 + ... + WnXn  
      + W vector (W0,W1,W2....Wn) 찾기   
        + MSE(Mean Squared Error)
          + (F(x)-y)^2 
 
       + 최소제곱법의 추정  
    + matrix 에 적용 w = A-1*B 적용
    + A : XtX and B : XtY
    + w = (XtX) -1 * (XtY)  
      + https://mazdah.tistory.com/831  
  
    + 다중 선형회귀 (정규방정식)
      + https://jangpiano-science.tistory.com/111
    
    + kernel regression matrix (일반화) 
      + (XtX) -1 가 존재하지 않을때 계산이 불가능한 단점
      + X : Kernel Trick mapping  
      + https://analysisbugs.tistory.com/163   
  
  + Another solving : Gradient descendant , Maximum likelyhood 
  
  + Model analysis: 
    + 1. Overfitting vs Generalization   
    + 2. evaluation : Training set and Test set 
    + 3. Validation set : K fold Validation + ETC  
      + https://deep-learning-study.tistory.com/623

+ 3주차 강의 
  + Square mean error vs Maximizes the probability 
  
  + Maximum likelyhood : (MLE) 
    +  데이터 D 가 가장 fit (maximize) 확률?
    +  weight 와 Standard Deviation 이 주어질 때 데이터(D)의 X,Y 좌표 : P(x,y|w,s)
        + Noise 가 정규분포를 띈다고 가정?   
        + Y와 X는 확률 곱 법칙으로 분리가능 
        + X 와 w,s 는 독립이다 (Conditional proabability) (chain rule 적용)  
        + 최종적으로 SLE와 MLE 는 같음 
        + https://ko.d2l.ai/chapter_deep-learning-basics/linear-regression.html
  
  + Logisitic regression 
        
    + 함수 
      + Sigmoid fuction : 나오는 값 0 ~ 1 사이 (확률값을 Regression 하려는 용도) 
    + 오즈비 
      + 승산 (성공확률 P / 실패확률 1-P) 
  
    + Logit Transform
      + log(odds) = log(p/1-p) = log(sigmoid p / 1-sigmoid) 
      + 로그 - 우도함수 (log likelywood function) 
      + 입력변수 1 단위 증가할때 log(odds) 의 변화량 
      + 회귀계수 W에 대해 미분 적용 
      
     
  + Naïve Bayesian
    
    + Naïve Bayesian Classifier
      + Probability with the strongest assumption on independence 
      + 조건부 확률 계산
        + https://ratsgo.github.io/machine%20learning/2017/05/18/naive/

    
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


+ Support Vector Machine
  + 직선을 하나 그려서 Boundary 를 지정하고 싶다. 어떤식으로 선을 그어야 할까?
  + 마진을 최대화 하는 Boundary !
  
  + 마진을 계산하는법?
     + 직선식 wtx+b =1 , wtx+b = 0 ,  wtx+b = -1  (wt : W transpose())
     + 두 직선사이의 거리 계산  
        + L2 norm 
          + |w| = (root)(w1^2+w2^2+ ...wn^2) = root (W*W_transpose())  
        
       + OX+ and OX- vector 가정 (원점에서 출발 직선위에 있는 벡터) 
       + 1. |X+ - X-| = M  (마진크기) 
       + 2.  X+ = X- + a * w  (a:상수) 
       + 1과 2식을 조합하면
         + |X+ - X-| = M  = a|w|
            + wt*x+b = 1 식에다가 x+ = x- + aw 대입 
            + wt(x- + a*w) + b = 1 
            + 전개하면 wt*x- + a*wt*w+b = 1
            + 정리하면 a = 2/wt*w (wtx- + b = -1 대입)
            + wt*w = |w| 이므로 
            + m = 2/|w|
   
   + |w|가 0으로 갈때 margin 이 max
     + min 0.5|w| 의 값을 계산하는 것
     + constraint(제약식) 추가 가능
   
    + 예시 : wx+b >1 , wx+b<-1 
    + 예시 : D = (1,1,-1) ,(2,2,+1)
      + min 0.5(w1^2+x2^2) 
      + subject to (w1+w2+b+1 <=0 and -2w1-2w2-b+1<=0)
        + F = 0.5(w1^2+x2^2) + a(w1+w2+b+1) + a2(-2w1-2w2-b+1)
    + 만족하는 식 :
      + x1 , x2에 대해서 각각 미분한식을 만족 + 4개의 추가 조건 획득  
        + case1 a1 = 0 a2 = 0  
        + case2 ,case3 case4 ... 에 따라서 case 별로 대입해보기 

+ Dual form SVM
  
  + L(W,b,a)
 
  + 식 0.5w*wt 에서
    + L(w,b,a)를 w에 대한 미분식 sigma (a_i * y_i * x_i)대입
      + 변형된 식 0.5*sigma_sigma(a_i*a_j*y_i*y_j*x*x_transpose)
      
  + Constraint 식 a_i(y_i(w_t*x_i+b)-1) 에서 
    + L(w,b,a) 를 b에 대한 미분식 sigma (a_i*y_i) = 0     
    + L(w,b,a) 를 w에 대한 미분식 w = sigma (a_i*y_i*x_i) 
      + 변형된 식 -sigma_sigma(a_i*a_j*y_i*y_j*x*x_transpose) + sigma a_i
  
  + 이제 a만 구하면 되는 dual form 으로 변형 완성

+ Soft-Margin SVM
  + 0.5|w|^2 + C*sigma ksi
  + ksi 를 도입 (slack variable / Training error 허용)
  
+ Non-linear SVM
  + 데이터를 고차원으로 mapping      
  + equality constraint , inequality constraint 
  + data X -> O(x) 로 변경 (예시 Xt*X -> O(X)t * O(x) 로 변형)

+ Kernel trick
+ https://ratsgo.github.io/machine%20learning/2017/05/30/SVM3/
  + 커널을 이용하여 mapping = inner product 줄이기? 
    + (x1*x2+1)^3 으로 간단하게 표헌가능 !
  + 간단하게 만드는 커널 Transform 형태 (연산수 줄이기)    
  + 커널은 뭐가 될수있어?
    + Mercer's condition 만족  

6주차 - 10/4(월) 

+ 의사 결정 나무 모델
  + 장점 : IF / Then Rule 생성 가능 
  + https://ratsgo.github.io/machine%20learning/2017/03/26/tree/
  
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
    + https://ratsgo.github.io/machine%20learning/2017/04/19/KC/
  
  + Gaussian Mixture
    + Randomly choose one of Gaussians, and generate a data
  
+ 8주차
  + PCA
  + https://ratsgo.github.io/machine%20learning/2017/04/24/PCA/
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
  
 + 순서요약
    + 데이터 정규화 (mean clustering)
    + 공분산 행렬 계산 
    + 공분산 행렬에서 eigenvalue 및 eigenvector 계산
    + 고유값 순서대로 나열 
    + 정렬된 고유값을 토대로 기존 변수를 변환
      + Z1 = e(1)X = e11*x1 + e12*x2 + ...
      + Z2 = e(2)X = e21*x1 + e22*x2 + ...
      + ...  
   
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
      + 2. 주어진 모델과 관측에 대해서 hidden State 의 Path? = Highest Prob. (maximize)
      + 3. 데이터가 주어졌을때(O) 가장 잘되는 모델 찾기 ? = Expectation and Maximize 
      
        + Evaluation , (신호가 어디에서 왔는가?) : 
        + Decoding , (Word 에서 형태소 POS 분석하는것)  : 
        + Learning or estimation  (A,B Pi 를 어떤식으로 학습을 하는가?)
 
   + A,B,Pi 
     + 1. Forward & Backward
       + https://ratsgo.github.io/speechbook/docs/am/hmm
     + 2. Viterbi 알고리즘
       + pass
     + 3. Baul Welch 알고리즘 
       + https://ratsgo.github.io/speechbook/docs/am/baumwelch
