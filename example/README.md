# Calibration Example Code

### 00_CameraCalibration.py
체커보드 인식해서 CameraCalibration하는 코드

### 01_StereoCalibration.py
Intrinsic을 알고 있는 카메라들과 동시에 촬영한 체커보드 이용해서 두 카메라간의 transformation matrix를 구하는 코드(현재 작성된 코드는 체커보드를 앞뒤로 붙여서 마주보고 있는 카메라를 캘리한 상황, 인덱스도 그에 맞춰서 변경되는 상황이라 확인 필요)

### 02_FindRect.py
카메라에서 바라본 네점과 그 네점의 3D 좌표를 이용해서 3D 좌표계의 카메라 위치를 획득하는 코드(지하철 창문의 4개의 꼭지점을 입력으로 넣었을때 카메라의 위치를 찾기위해 작성한 코드)

### 03~99 까지는 디버깅용 코드라 일단 무시
