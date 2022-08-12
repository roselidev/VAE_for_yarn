yarn(thread) 실 하나의 올풀림/뭉침/끊김 현상을 감지하기 위한 모델

가정 : 대량의 정상 실 데이터로 오토인코더를 학습시키고 -> 이상 실 데이터를 넣으면 Anomaly Score가 치솟을 것이다.

실험결과 : Anomaly Score 유의미한 변화가 없어 실패

실패원인 : 실의 올풀림 등은 이미지 변화량이 너무 적음

추후실험 : ROI를 박스모양 대신 polygon, polyline으로 구성해볼 수 있음. 그러나 실이 ROI 안에서 움직이는 usecase이기 때문에 thread detection module을 추가하여야함. 실시간 저메모리 케이스에 맞지 않음.