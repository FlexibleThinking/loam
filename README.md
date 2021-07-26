## Branch
##### [브랜치 사용법](http://amazingguni.github.io/blog/2016/03/git-branch-%EA%B7%9C%EC%B9%99)
- Local branch :: remote에 올리지 않고 각자 작업
  - develop branch에서 local branch 분기
  - ex) `git checkout -b local`
  - develop에 작업 내용 합치고 싶은 경우 `git merge --no-ff develop`
- Develop branch :: 실질적으로 개발에 사용하는 branch
- release branch :: develop을 master와 합치기 전 임시로 만드는 branch
  - 여기서 버그 픽스
  - 버그 픽스가 끝나면 `git merge --no-ff master`를 통해 merge
  - 이 때 tagging 필수
  - master에 merge 이후 develop으로도 merge
  - 이후 release branch 삭제
## Profiling
- `std::cout<<"Process() start"<<std::endl;`추가
  - terminal에서 찍히는 것을 확인

## CMakefiles
  - CUDA compiler위치 명시.
  - project language CXX CUDA 전 set으로 compiler위치 명시 및 c++ 17 사용 선언
 
