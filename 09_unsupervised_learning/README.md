# 비지도 학습 
- 레이블이 없는 상태에서 학습을 하는 것.
- 이제 막 발을 담그기 시작한 분야 
- clustering, outtlier detection, density estimation에 쓰이곤 함.

## 예시
![](images/unsupervised_learning/classification_vs_clustering_plot.png)

![](images/unsupervised_learning/clustering_example.png)

# clustering
1. K Means
2. DBSCAN

## K Means
K개의 클러스터로 만들기

1. 랜덤으로 센트로이드 할당
2. voronoi diagram 생성
3. 센트로이드 업테이트 
4. 2~3 반복 
(voronoi diagram: 평면을 특정 점까지의 거리가 가장 가까운 점의 집합으로 분할한 그림

`R={x in X | d(x, P_k) <= d(x, P_j) j!= k`

P_k = X의 모든 점들
)
![](images/unsupervised_learning/kmeans_algorithm_plot.png)

## 평가
이녀셔 inertia: 각 샘플과 가장 가까운 센트로이드 사이의 평균 제곱 거리

## 특징
제한된 횟수 안에 수렴하지만 local optimum임. 매 실행시 다른 결과가 나올 수도 있다.

## KMeans sklearn
- 이레 논문에 사용된 알고리즘들이 KMeans에 기본값으로 구현되어 있음.
- K Means는 초기에 램덤하게 센트로이드를 지정하는 것이 중요하다. 이것을 다룬 논문이 2006 [KMeans++](http://ilpubs.stanford.edu:8090/778/1/2006-13.pdf)
    - KMeans++
        1. 무작위로 센트로이드 하나를 선택한다.
        2. 위에서 선택한 센트로이드에서 멀리 떨어진 센트로이드를 환률적으로 선택한다.
        3. 반복
- KMeans 알고리즘에서 불필요한 계산을 줄인 논문 2014 [Using the Triangle Inequality to Accelerate K-Means](https://www.researchgate.net/publication/2480121_Using_the_Triangle_Inequality_to_Accelerate_K-Means)
- 기계학습은 보통 큰 데이터셋으로 진행되기 떄문에 메모리 문제를 야기함. 미니배치로 학습하는 논문이 나옴. 2010 [Web-scale k-means clustering](https://dl.acm.org/doi/10.1145/1772690.1772862)

# K 찾기
- KMeans는 적절한 K를 알아야한다는 단점이 있다. 데이터가 N 차원이면 사실 알 수 없음
- 적당한 K를 찾는 방법으로 아래와 같은 방법이 있다.
1. Elbow
    - 이너셔가 낮으면 대체로 좋은 모델일 것이라고 추측
    - 이너셔 감소의 기울가 변하는 지점을 최적의 K로 본다.
    - 좋은 방법이 아닐때가 많다.
![](images/unsupervised_learning/inertia_vs_k_plot.png)
2. Silhouuette 계수
    - 클러스터에 데이터가 적절히 분포되었다면 좋은 모델일거라고 추측
    - 그래프의 너비: 클러스터가 가진 샘플의 갯수.
    - 그래프의 높이: 실루엣 계수 (샘플이 클러스터에 잘 속해 있는건 지를 판단하는 측정치)
    - 아래 예시에서는 클러스터가 적절히 분배되고 일정 실루엣 계수 이상을 가진 k=5가 가장 적합.
![](images/unsupervised_learning/silhouette_analysis_plot.png)
