# For DDP(Distributed Data Parallel) in pytorch

- Pytorch 1.6 버전부터 **[torch.distributed](https://pytorch.org/docs/stable/distributed.html)** 라이브러리에 3개의 요소로 구성됨
    1. Distributed Data-Parallel Training(DDP)
        - 단일 프로그램 다중 데이터 학습 패러다임
        - 모델(신경망)은 모든 프로세서에 복제되고, 모든 복제 모델은 서로 다른 데이터 샘플을 입력받음
        - 복제 모델간의 동기화 유지를 위해 미분값에 대한 통신을 처리
        - 교육속도를 높이기 위해 미분값 계산 후, 미분값 중첩
    2. RPC-Based Distributed Training(RPC)
        - Data-parallel 학습을 지원할 수 없는 일반적인 학습 구조에서 활용
        - Distributed pipeline parallelism
        - Parameter server paradigm
        - DDP와 다른 학습 패러다임 조합
        - 원격으로 객체(프로그램)의 Lifetime 관리 및 머신의 경계를 넘어선 Autograd engine 확장을 도움
    3. Collective Communication(c10d)
        - Group내 프로세서들 간 Tensor 통신(DDP와 RPC의 기반)
        - Collective 통신(DDP 사용): all_reduce, all_gather
        - P2P 통신(RPC 사용): send, isend
        - DDP와 RPC API는 대다수의 Distributed 학습 방식을 제공하기 때문에, 개발자가 직접적으로 활용하는 경우는 드묾.
        - 하지만, 직접적인 API 호출도 유용함(Distributed parameter averaging)
        - 프로그램에서 DDP를 활용한 미분값 전달하는 대신, 역전파(backward) 후에 모든 모델의 파라미터 값의 평균을 계산.
        - 계산과 통신을 분리할 수 있음
        - 통신 대상을 보다 세밀하게 제어 가능함 
        - DDP가 제공하는 최적화 성능은 포기해야됨

[My Notion Link](https://dy-research.notion.site/Pytorch-DDP-DistributedDataParallel-045b146d1689455aa1b411c30b8a782b?pvs=4)

## Data Parallel Training

1. 단일 GPU 학습 (데이터와 모델이 단일 GPU에 할당 가능한 경우, 속도는 관심없음)
2. 단일 서버(기계) & 다중 GPU 학습 (DataParallel)
    - 속도 증가 및 최소한의 코드 수정
    [Pytorch Distributed 정리 - I](https://www.notion.so/Pytorch-Distributed-I-15147c02da5749d285ba0c438d1a055f?pvs=21)
    
3. 단일 서버(기계) & 다중 GPU 학습 (DistributedDataParallel)
    - 속도 증가 및 코드 수정 불가피
    [Pytorch Distributed 정리 - II](https://www.notion.so/Pytorch-Distributed-II-328d5557c91149dcab3f078d60d7c0cd?pvs=21)
    [Pytorch Distributed 정리 - III](https://www.notion.so/Pytorch-Distributed-III-e259eade397f4392be7c63188f9bb7e9?pvs=21)
    [Pytorch Distributed 정리 - IV](https://www.notion.so/Pytorch-Distributed-IV-0110a3c7228a429fb7c83d15a2c81109?pvs=21)
    
4. 다중 서버(기계) (DistributedDataParallel & launching script) 
& 분산 학습(torch.distributed.elastic → torchrun)
    - 모델의 규모가 너무 커서 서버의 경계를 넘나들어야될 경우
    - 오류가 예상되거나 학습중에 리소스가 동적으로 연결 및 종료 될 경우
